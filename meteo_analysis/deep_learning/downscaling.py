"""Orography-aware, hard-consistent super-resolution for weather fields."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .fronts import ResidualBlock, StandardizeWithValidity


def block_mean(values, scale: int, area_weights=None):
    if area_weights is None:
        return F.avg_pool2d(values, kernel_size=scale, stride=scale)
    weights = torch.broadcast_to(area_weights, values.shape)
    numerator = F.avg_pool2d(
        values * weights, kernel_size=scale, stride=scale
    )
    denominator = F.avg_pool2d(
        weights, kernel_size=scale, stride=scale
    )
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny)


def _repeat_blocks(values, scale: int):
    return values.repeat_interleave(scale, -2).repeat_interleave(scale, -1)


def enforce_block_mean(candidate, coarse_reference, scale: int, area_weights=None):
    """Project each high-resolution block onto an exact coarse-cell mean."""
    correction = coarse_reference - block_mean(candidate, scale, area_weights)
    return candidate + _repeat_blocks(correction, scale)


def enforce_nonnegative_block_mean(
    candidate, coarse_reference, scale: int, area_weights=None
):
    """Guarantee non-negativity and an exact coarse-cell spatial mean."""
    positive = F.softplus(candidate)
    mean = block_mean(positive, scale, area_weights)
    reference = coarse_reference.clamp_min(0.0)
    factor = torch.where(
        reference > 0,
        reference / mean.clamp_min(torch.finfo(mean.dtype).tiny),
        torch.zeros_like(reference),
    )
    return positive * _repeat_blocks(factor, scale)


class OrographicDownscaler(nn.Module):
    """Residual x4-style downscaler with optional coarse-scale bias correction.

    The high-resolution output is a redistribution of ``corrected_coarse``.
    Temperature and wind preserve the block mean; precipitation is additionally
    non-negative.  This separates large-scale calibration from spatial detail.
    """

    def __init__(
        self,
        *,
        coarse_mean,
        coarse_standard_deviation,
        static_mean,
        static_standard_deviation,
        output_coarse_indices=(0, 1, 2, 3),
        precipitation_index=1,
        scale=4,
        base_channels=48,
        residual_blocks=6,
        bias_correction=True,
    ):
        super().__init__()
        scale = int(scale)
        if scale < 2 or scale & (scale - 1):
            raise ValueError("scale deve essere una potenza di due >= 2")
        self.scale = scale
        self.output_coarse_indices = tuple(map(int, output_coarse_indices))
        self.output_channels = len(self.output_coarse_indices)
        if not 0 <= precipitation_index < self.output_channels:
            raise ValueError("indice precipitazione non valido")
        self.precipitation_index = int(precipitation_index)
        self.bias_correction = bool(bias_correction)
        self.coarse_norm = StandardizeWithValidity(
            coarse_mean, coarse_standard_deviation
        )
        self.static_norm = StandardizeWithValidity(
            static_mean, static_standard_deviation
        )
        self.coarse_stem = nn.Conv2d(
            self.coarse_norm.channels * 2, base_channels, 3, padding=1
        )
        self.coarse_body = nn.Sequential(*[
            ResidualBlock(base_channels, base_channels)
            for _ in range(int(residual_blocks))
        ])
        self.bias_head = nn.Conv2d(base_channels, self.output_channels, 1)
        # The untrained network must reproduce the native coarse field, not
        # introduce an arbitrary large-scale correction at initialisation.
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)
        upsamplers = []
        for _ in range(int(math.log2(scale))):
            upsamplers.extend((
                nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
            ))
        self.upsample = nn.Sequential(*upsamplers)
        static_channels = max(16, base_channels // 2)
        self.static_stem = nn.Sequential(
            nn.Conv2d(
                self.static_norm.channels * 2, static_channels, 3, padding=1
            ),
            nn.SiLU(inplace=True),
            ResidualBlock(static_channels, static_channels),
        )
        fused = base_channels + static_channels + self.output_channels
        self.refine = nn.Sequential(
            ResidualBlock(fused, base_channels),
            ResidualBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, self.output_channels, 3, padding=1),
        )

    def _coarse_reference(self, coarse, encoded):
        indices = torch.as_tensor(
            self.output_coarse_indices, device=coarse.device, dtype=torch.long
        )
        reference = torch.index_select(coarse, 1, indices)
        if self.bias_correction:
            reference = reference + self.bias_head(encoded)
        # Precipitation cannot be negative, including after bias correction.
        rain = reference[
            :, self.precipitation_index:self.precipitation_index + 1
        ].clamp_min(0.0)
        return torch.cat((
            reference[:, :self.precipitation_index], rain,
            reference[:, self.precipitation_index + 1:],
        ), dim=1)

    def forward(self, coarse, static, cell_area_m2):
        if static.shape[-2:] != (
            coarse.shape[-2] * self.scale, coarse.shape[-1] * self.scale
        ):
            raise ValueError("griglia statica non allineata al fattore di scala")
        area_weights = cell_area_m2
        if area_weights.ndim == 3:
            area_weights = area_weights[:, None]
        if area_weights.shape != (coarse.shape[0], 1, *static.shape[-2:]):
            raise ValueError("area delle celle non allineata alla griglia fine")
        if not torch.all(torch.isfinite(area_weights) & (area_weights > 0)):
            raise ValueError("area delle celle non finita o non positiva")
        encoded = F.silu(self.coarse_stem(self.coarse_norm(coarse)))
        encoded = self.coarse_body(encoded) + encoded
        corrected = self._coarse_reference(coarse, encoded)
        high_features = self.upsample(encoded)
        static_features = self.static_stem(self.static_norm(static))
        baseline = _repeat_blocks(corrected, self.scale)
        candidate = baseline + self.refine(torch.cat((
            high_features, static_features, baseline,
        ), dim=1))

        before = candidate[:, :self.precipitation_index]
        rain = candidate[:, self.precipitation_index:self.precipitation_index + 1]
        after = candidate[:, self.precipitation_index + 1:]
        constrained = []
        if before.shape[1]:
            constrained.append(enforce_block_mean(
                before, corrected[:, :self.precipitation_index], self.scale,
                area_weights,
            ))
        constrained.append(enforce_nonnegative_block_mean(
            rain,
            corrected[:, self.precipitation_index:self.precipitation_index + 1],
            self.scale, area_weights,
        ))
        if after.shape[1]:
            constrained.append(enforce_block_mean(
                after, corrected[:, self.precipitation_index + 1:], self.scale,
                area_weights,
            ))
        prediction = torch.cat(constrained, dim=1)
        return {
            "prediction": prediction,
            "corrected_coarse": corrected,
            "area_weights": area_weights,
        }


class DownscalingLoss(nn.Module):
    """Variable-aware field, gradient and coarse-calibration objectives."""

    def __init__(
        self,
        *,
        precipitation_index=1,
        channel_weights=(1.0, 2.0, 1.0, 1.0),
        gradient_weight=0.15,
        coarse_bias_weight=0.35,
    ):
        super().__init__()
        self.precipitation_index = int(precipitation_index)
        self.register_buffer(
            "channel_weights", torch.as_tensor(channel_weights, dtype=torch.float32)
        )
        self.gradient_weight = float(gradient_weight)
        self.coarse_bias_weight = float(coarse_bias_weight)

    @staticmethod
    def _masked_mean(values, valid):
        authority = valid.to(values.dtype)
        return (values * authority).sum() / authority.sum().clamp_min(1.0)

    def forward(self, outputs, target, valid_mask=None):
        prediction = outputs["prediction"]
        if prediction.shape != target.shape:
            raise ValueError("previsione e target downscaling non allineati")
        if prediction.shape[1] != len(self.channel_weights):
            raise ValueError("numero pesi canale non coerente")
        valid = (
            torch.isfinite(target) if valid_mask is None
            else valid_mask.bool() & torch.isfinite(target)
        )
        safe_target = torch.where(valid, target, prediction.detach())
        losses = []
        for channel in range(prediction.shape[1]):
            predicted_channel = prediction[:, channel]
            target_channel = safe_target[:, channel]
            if channel == self.precipitation_index:
                error = F.smooth_l1_loss(
                    torch.log1p(predicted_channel.clamp_min(0.0)),
                    torch.log1p(target_channel.clamp_min(0.0)),
                    reduction="none",
                )
            else:
                error = F.smooth_l1_loss(
                    predicted_channel, target_channel, reduction="none"
                )
            losses.append(self._masked_mean(error, valid[:, channel]))
        field_loss = torch.sum(torch.stack(losses) * self.channel_weights)

        gradient_terms = []
        for axis in (-2, -1):
            predicted_gradient = torch.diff(prediction, dim=axis)
            target_gradient = torch.diff(safe_target, dim=axis)
            pair_valid = torch.diff(valid.to(torch.int8), dim=axis) == 0
            if axis == -2:
                pair_valid &= valid[:, :, 1:, :] & valid[:, :, :-1, :]
            else:
                pair_valid &= valid[:, :, :, 1:] & valid[:, :, :, :-1]
            gradient_terms.append(self._masked_mean(
                torch.abs(predicted_gradient - target_gradient), pair_valid
            ))
        gradient_loss = torch.stack(gradient_terms).mean()

        scale = prediction.shape[-1] // outputs["corrected_coarse"].shape[-1]
        area_weights = outputs.get("area_weights")
        coarse_target = block_mean(safe_target, scale, area_weights)
        coarse_valid = block_mean(valid.to(prediction.dtype), scale) >= 0.999
        coarse_error = F.smooth_l1_loss(
            outputs["corrected_coarse"], coarse_target, reduction="none"
        )
        coarse_loss = self._masked_mean(coarse_error, coarse_valid)
        total = (
            field_loss + self.gradient_weight * gradient_loss
            + self.coarse_bias_weight * coarse_loss
        )
        return {
            "loss": total,
            "field": field_loss.detach(),
            "gradient": gradient_loss.detach(),
            "coarse_bias": coarse_loss.detach(),
        }
