"""Residual U-Net and loss functions for multi-class weather fronts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class StandardizeWithValidity(nn.Module):
    """Normalise physical channels and expose missingness to the network."""

    def __init__(self, mean, standard_deviation):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(standard_deviation, dtype=torch.float32)
        if mean.ndim != 1 or std.shape != mean.shape or torch.any(std <= 0):
            raise ValueError("statistiche di normalizzazione non valide")
        self.register_buffer("mean", mean[None, :, None, None])
        self.register_buffer("std", std[None, :, None, None])

    @property
    def channels(self):
        return int(self.mean.shape[1])

    def forward(self, values):
        if values.ndim != 4 or values.shape[1] != self.channels:
            raise ValueError("tensore incompatibile con lo schema dei canali")
        valid = torch.isfinite(values)
        normal = torch.where(valid, (values - self.mean) / self.std, 0.0)
        return torch.cat((normal, valid.to(normal.dtype)), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, values):
        return F.silu(self.body(values) + self.skip(values), inplace=True)


class FrontUNet(nn.Module):
    """U-Net with explicit front-existence and multi-class output heads."""

    def __init__(
        self,
        *,
        input_mean,
        input_standard_deviation,
        class_count: int = 5,
        base_channels: int = 32,
    ):
        super().__init__()
        if class_count < 2:
            raise ValueError("servono background e almeno una classe frontale")
        self.standardize = StandardizeWithValidity(
            input_mean, input_standard_deviation
        )
        channels = [base_channels * (2 ** index) for index in range(4)]
        self.encoders = nn.ModuleList()
        current = self.standardize.channels * 2
        for output in channels:
            self.encoders.append(ResidualBlock(current, output))
            current = output
        self.bottleneck = ResidualBlock(channels[-1], channels[-1] * 2)
        self.decoders = nn.ModuleList()
        current = channels[-1] * 2
        for skip in reversed(channels):
            self.decoders.append(ResidualBlock(current + skip, skip))
            current = skip
        self.class_head = nn.Conv2d(current, class_count, 1)
        self.frontness_head = nn.Conv2d(current, 1, 1)
        self.class_count = int(class_count)

    def forward(self, inputs):
        values = self.standardize(inputs)
        skips = []
        for encoder in self.encoders:
            values = encoder(values)
            skips.append(values)
            values = F.max_pool2d(values, 2)
        values = self.bottleneck(values)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            values = F.interpolate(
                values, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            values = decoder(torch.cat((values, skip), dim=1))
        return {
            "class_logits": self.class_head(values),
            "frontness_logits": self.frontness_head(values)[:, 0],
        }


class FrontSegmentationLoss(nn.Module):
    """Class-balanced CE + Dice + hierarchical front-existence supervision."""

    def __init__(
        self,
        class_weights=None,
        *,
        cross_entropy_weight=1.0,
        dice_weight=0.7,
        frontness_weight=0.25,
        contradiction_weight=0.05,
        ignore_index=-100,
    ):
        super().__init__()
        weights = None if class_weights is None else torch.as_tensor(
            class_weights, dtype=torch.float32
        )
        self.register_buffer("class_weights", weights)
        self.ce_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)
        self.frontness_weight = float(frontness_weight)
        self.contradiction_weight = float(contradiction_weight)
        self.ignore_index = int(ignore_index)

    @staticmethod
    def _weighted_mean(values, weights, valid):
        authority = torch.where(valid, weights, torch.zeros_like(weights))
        return (values * authority).sum() / authority.sum().clamp_min(1.0)

    def forward(
        self,
        outputs,
        target,
        *,
        label_weight=None,
        physics_support=None,
    ):
        logits = outputs["class_logits"]
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("logits e target frontali non allineati")
        valid = target != self.ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        weights = (
            torch.ones_like(target, dtype=logits.dtype)
            if label_weight is None else label_weight.to(logits.dtype)
        )
        ce = F.cross_entropy(
            logits, target, weight=self.class_weights,
            ignore_index=self.ignore_index, reduction="none",
        )
        ce = self._weighted_mean(ce, weights, valid)

        probability = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(
            safe_target.clamp(0, logits.shape[1] - 1), logits.shape[1]
        ).permute(0, 3, 1, 2).to(probability.dtype)
        authority = (weights * valid).unsqueeze(1)
        intersection = (probability * target_one_hot * authority).sum((0, 2, 3))
        denominator = ((probability + target_one_hot) * authority).sum((0, 2, 3))
        # Dice is evaluated on frontal classes only; background dominance
        # otherwise makes an empty-looking mask score deceptively well.
        dice = 1.0 - ((2.0 * intersection[1:] + 1.0) /
                      (denominator[1:] + 1.0)).mean()

        front_target = (safe_target > 0).to(logits.dtype)
        front_bce = F.binary_cross_entropy_with_logits(
            outputs["frontness_logits"], front_target, reduction="none"
        )
        front_bce = self._weighted_mean(front_bce, weights, valid)

        contradiction = logits.new_zeros(())
        if physics_support is not None and self.contradiction_weight > 0:
            support = physics_support.to(logits.dtype).clamp(0.0, 1.0)
            front_probability = 1.0 - probability[:, 0]
            contradiction = self._weighted_mean(
                front_probability * (1.0 - support), weights, valid
            )
        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.frontness_weight * front_bce
            + self.contradiction_weight * contradiction
        )
        return {
            "loss": total,
            "cross_entropy": ce.detach(),
            "dice": dice.detach(),
            "frontness": front_bce.detach(),
            "physics_contradiction": contradiction.detach(),
        }
