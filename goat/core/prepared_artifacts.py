from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch


PREPARED_ARTIFACT_SCHEMA = 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return the stable JSON representation used in artifact fingerprints."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def metadata_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _update_array_hash(hasher, value: Any, *, name: str) -> None:
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(canonical_json({"shape": list(array.shape)}).encode("ascii"))
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), _HASH_CHUNK_BYTES):
        hasher.update(raw[start : start + _HASH_CHUNK_BYTES])


def fingerprint_encoded_dataset(dataset, *, label_fields: Iterable[str] = ()) -> str:
    """Hash immutable encoded features and, optionally, selected label fields.

    Raw EM fits depend only on ``data``. Geometry artifacts may additionally name
    label fields. A full content digest is intentionally used: object ids, paths,
    and mtimes are not reliable identities for long-running experiment sweeps.
    """

    if not hasattr(dataset, "data"):
        raise TypeError("encoded dataset must expose a .data field")
    hasher = hashlib.sha256()
    _update_array_hash(hasher, dataset.data, name="data")
    for field in label_fields:
        value = getattr(dataset, field, None)
        if value is None:
            hasher.update(f"{field}=none".encode("utf-8"))
        else:
            _update_array_hash(hasher, value, name=field)
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(_HASH_CHUNK_BYTES)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


class PreparedArtifactStore:
    """Versioned, checksum-validated, process-safe prepared artifact storage."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser().resolve()

    def key(self, metadata: Mapping[str, Any]) -> str:
        envelope = {
            "schema_version": PREPARED_ARTIFACT_SCHEMA,
            "metadata": dict(metadata),
        }
        return metadata_fingerprint(envelope)

    def _paths(self, kind: str, key: str) -> tuple[Path, Path, Path]:
        base = self.root / str(kind) / key
        return base.with_suffix(".pt"), base.with_suffix(".json"), base.with_suffix(".lock")

    def paths(self, kind: str, key: str) -> tuple[Path, Path, Path]:
        """Return payload, manifest, and lock paths for an artifact key."""

        return self._paths(kind, key)

    @contextmanager
    def lock(self, kind: str, key: str) -> Iterator[None]:
        """Serialize preparation of one key while allowing independent keys."""

        import fcntl

        payload_path, _, lock_path = self._paths(kind, key)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def load(
        self,
        kind: str,
        metadata: Mapping[str, Any],
        *,
        required: bool = False,
    ) -> Any | None:
        key = self.key(metadata)
        payload_path, manifest_path, _ = self._paths(kind, key)
        if not payload_path.exists() or not manifest_path.exists():
            if required:
                raise FileNotFoundError(
                    f"required prepared artifact is missing: kind={kind} key={key}"
                )
            return None

        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            if required:
                raise RuntimeError(f"invalid prepared-artifact manifest: {manifest_path}") from exc
            return None

        expected_metadata_sha = metadata_fingerprint(dict(metadata))
        valid = (
            int(manifest.get("schema_version", -1)) == PREPARED_ARTIFACT_SCHEMA
            and manifest.get("kind") == kind
            and manifest.get("key") == key
            and manifest.get("metadata_sha256") == expected_metadata_sha
            and manifest.get("payload_sha256") == file_sha256(payload_path)
        )
        if not valid:
            if required:
                raise RuntimeError(f"prepared artifact failed validation: {payload_path}")
            return None
        return torch.load(payload_path, map_location="cpu", weights_only=False)

    def save(self, kind: str, metadata: Mapping[str, Any], payload: Any) -> Path:
        key = self.key(metadata)
        payload_path, manifest_path, _ = self._paths(kind, key)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        nonce = f"{os.getpid()}.{os.urandom(6).hex()}"
        payload_tmp = payload_path.with_name(f".{payload_path.name}.{nonce}.tmp")
        manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{nonce}.tmp")
        try:
            torch.save(payload, payload_tmp)
            payload_sha = file_sha256(payload_tmp)
            manifest = {
                "schema_version": PREPARED_ARTIFACT_SCHEMA,
                "kind": kind,
                "key": key,
                "metadata": dict(metadata),
                "metadata_sha256": metadata_fingerprint(dict(metadata)),
                "payload_sha256": payload_sha,
            }
            manifest_tmp.write_text(canonical_json(manifest) + "\n")
            os.replace(payload_tmp, payload_path)
            os.replace(manifest_tmp, manifest_path)
        finally:
            payload_tmp.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
        return payload_path
