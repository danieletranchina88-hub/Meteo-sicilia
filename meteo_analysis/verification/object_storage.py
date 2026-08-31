"""Immutable S3-compatible retention for full-resolution ICON source GRIBs.

The public GitHub Pages snapshot deliberately contains derived products, not
raw GRIB.  This module retains the exact input files in a private object store
when explicitly enabled by the pipeline environment.  It is compatible with
AWS S3 and S3 APIs such as Cloudflare R2 and Backblaze B2.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from .archive import sha256_file


SCHEMA_VERSION = 1
MODE_ENV = "ICON_ARCHIVE_MODE"
MODE_RAW = "raw"
MODE_OFF = "off"


class ObjectStoreConfigurationError(RuntimeError):
    """Raised when a requested immutable archive is not safely configured."""


class ObjectStoreIntegrityError(RuntimeError):
    """Raised when a remote object differs from the local verified GRIB."""


def _run_tag(run_time: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "", str(run_time))
    if len(value) < 8:
        raise ValueError(f"run time non valido per archivio: {run_time!r}")
    return value


def _safe_fragment(value: str, *, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return (clean or fallback)[:128]


def _safe_prefix(value: str) -> str:
    parts = [
        _safe_fragment(part, fallback="archive")
        for part in str(value).split("/")
        if str(part).strip()
    ]
    return "/".join(parts) or "runs/icon2i"


@dataclass(frozen=True)
class ObjectStoreSettings:
    """Non-secret object storage settings read from the workflow environment."""

    mode: str
    bucket: str | None
    endpoint_url: str | None
    region_name: str | None
    access_key_id: str | None
    secret_access_key: str | None
    session_token: str | None
    prefix: str

    @property
    def enabled(self) -> bool:
        return self.mode == MODE_RAW

    @classmethod
    def from_environment(cls, environ=None) -> "ObjectStoreSettings":
        environment = os.environ if environ is None else environ
        mode = str(environment.get(MODE_ENV, MODE_OFF)).strip().lower() or MODE_OFF
        if mode not in {MODE_OFF, MODE_RAW}:
            raise ObjectStoreConfigurationError(
                f"{MODE_ENV} deve essere '{MODE_OFF}' o '{MODE_RAW}', non {mode!r}"
            )
        settings = cls(
            mode=mode,
            bucket=(environment.get("ICON_ARCHIVE_S3_BUCKET") or "").strip() or None,
            endpoint_url=(environment.get("ICON_ARCHIVE_S3_ENDPOINT") or "").strip() or None,
            region_name=(environment.get("ICON_ARCHIVE_S3_REGION") or "").strip() or None,
            access_key_id=(environment.get("ICON_ARCHIVE_S3_ACCESS_KEY_ID") or "").strip() or None,
            secret_access_key=(environment.get("ICON_ARCHIVE_S3_SECRET_ACCESS_KEY") or "").strip() or None,
            session_token=(environment.get("ICON_ARCHIVE_S3_SESSION_TOKEN") or "").strip() or None,
            prefix=_safe_prefix(environment.get("ICON_ARCHIVE_S3_PREFIX", "runs/icon2i")),
        )
        if settings.enabled:
            missing = [
                name for name, value in {
                    "ICON_ARCHIVE_S3_BUCKET": settings.bucket,
                    "ICON_ARCHIVE_S3_ENDPOINT": settings.endpoint_url,
                    "ICON_ARCHIVE_S3_ACCESS_KEY_ID": settings.access_key_id,
                    "ICON_ARCHIVE_S3_SECRET_ACCESS_KEY": settings.secret_access_key,
                }.items() if not value
            ]
            if missing:
                raise ObjectStoreConfigurationError(
                    "archivio raw richiesto ma mancano: " + ", ".join(missing)
                )
        return settings

    def public_status(self) -> dict[str, Any]:
        """Return a manifest-safe status without secrets, endpoint or bucket."""

        return {
            "schemaVersion": SCHEMA_VERSION,
            "mode": self.mode,
            "kind": "private-s3-compatible-raw-grib" if self.enabled else None,
            "immutability": (
                "content-addressed-key-and-sha256-head-verification"
                if self.enabled else None
            ),
            "retention": "external-object-store" if self.enabled else None,
        }


class RawGribArchive:
    """Write-once content-addressed retention for a single ICON run."""

    def __init__(self, settings: ObjectStoreSettings, *, run_time: str, client=None):
        self.settings = settings
        self.run_time = str(run_time)
        self.run_tag = _run_tag(run_time)
        self._client = client

    @classmethod
    def from_environment(cls, *, run_time: str, environ=None, client=None):
        return cls(ObjectStoreSettings.from_environment(environ), run_time=run_time, client=client)

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def public_status(self) -> dict[str, Any]:
        status = self.settings.public_status()
        status["runPrefix"] = (
            f"{self.settings.prefix}/{self.run_tag}/raw" if self.enabled else None
        )
        return status

    def _s3_client(self):
        if not self.enabled:
            raise ObjectStoreConfigurationError("archivio raw non abilitato")
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:  # pragma: no cover - CI installs boto3
            raise ObjectStoreConfigurationError(
                "boto3 non installato: impossibile usare lo storage S3"
            ) from error
        self._client = boto3.client(
            "s3",
            endpoint_url=self.settings.endpoint_url,
            region_name=self.settings.region_name or "auto",
            aws_access_key_id=self.settings.access_key_id,
            aws_secret_access_key=self.settings.secret_access_key,
            aws_session_token=self.settings.session_token,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 4, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        )
        return self._client

    @staticmethod
    def _not_found(error: Exception) -> bool:
        response = getattr(error, "response", {}) or {}
        code = str((response.get("Error") or {}).get("Code") or "")
        return code in {"404", "NoSuchKey", "NotFound"}

    def _key(self, *, path: Path, role: str, sha256: str) -> str:
        role_fragment = _safe_fragment(role, fallback="source")
        name_fragment = _safe_fragment(path.name, fallback="source.grib")
        return (
            f"{self.settings.prefix}/{self.run_tag}/raw/{role_fragment}/"
            f"{sha256[:20]}-{name_fragment}"
        )

    def retain_source(self, *, path, name: str, role: str, source_url: str,
                      expected_sha256: str | None = None) -> dict[str, Any] | None:
        """Upload one local GRIB exactly once, then verify remote identity.

        A pre-existing object is accepted only if both byte size and SHA-256
        metadata match.  Different content at the same immutable key fails the
        run instead of silently replacing history.
        """

        if not self.enabled:
            return None
        local_path = Path(path)
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        size_bytes = int(local_path.stat().st_size)
        sha256 = expected_sha256 or sha256_file(local_path)
        if not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise ObjectStoreIntegrityError("SHA-256 locale non valido")
        key = self._key(path=local_path, role=role, sha256=sha256)
        client = self._s3_client()
        existing = None
        try:
            existing = client.head_object(Bucket=self.settings.bucket, Key=key)
        except Exception as error:
            if not self._not_found(error):
                raise
        if existing is None:
            url_digest = hashlib.sha256(str(source_url).encode("utf-8")).hexdigest()
            client.upload_file(
                str(local_path), self.settings.bucket, key,
                ExtraArgs={
                    "Metadata": {
                        "sha256": sha256,
                        "source-url-sha256": url_digest,
                        "source-name": _safe_fragment(name, fallback="source"),
                        "run-time": self.run_time,
                    },
                },
            )
            existing = client.head_object(Bucket=self.settings.bucket, Key=key)

        remote_size = int(existing.get("ContentLength") or -1)
        remote_metadata = {
            str(key_name).lower(): str(value)
            for key_name, value in (existing.get("Metadata") or {}).items()
        }
        if remote_size != size_bytes or remote_metadata.get("sha256") != sha256:
            raise ObjectStoreIntegrityError(
                "oggetto S3 esistente non coincide con il GRIB verificato: "
                f"{key}"
            )
        return {
            "key": key,
            "sizeBytes": size_bytes,
            "sha256": sha256,
            "retention": "private-s3-compatible-object-store",
            "versionId": existing.get("VersionId") or None,
            "etag": str(existing.get("ETag") or "").strip('"') or None,
        }
