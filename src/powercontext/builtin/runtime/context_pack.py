# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunked Context Pack wire format with prefix reuse deltas."""

# ruff: noqa: TRY003 - every protocol error carries its own diagnostic text.
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class ContextPackError(ValueError):
    """A Context Pack or delta violated the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context body split into validated byte chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a complete pack from already chunked bytes."""

        chunks = tuple(bytes(item) for item in items)
        digest = _pack_digest(chunks)
        # byte_count counts the serialized form, which includes byte_count itself.
        size = 0
        for _ in range(8):
            candidate = cls(chunks, digest, size)
            measured = len(candidate.to_bytes())
            if measured == size:
                break
            size = measured
        return cls(chunks, digest, size)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split content deterministically into UTF-8 safe chunks."""

        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        data = content.encode("utf-8")
        items: list[bytes] = []
        start = 0
        while start < len(data):
            end = min(start + chunk_bytes, len(data))
            while end > start:
                try:
                    data[start:end].decode("utf-8")
                except UnicodeDecodeError:
                    end -= 1
                else:
                    break
            if end == start:
                raise ContextPackError("chunk_bytes cannot hold one UTF-8 character")
            items.append(data[start:end])
            start = end
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        """Serialize the pack to canonical wire JSON."""

        return _canonical({
            "schema": CONTEXT_PACK_SCHEMA,
            "items": [_encode(item) for item in self.items],
            "digest": self.digest,
            "byte_count": self.byte_count,
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and verify one serialized pack."""

        payload = _load_object(data, _PACK_FIELDS, "context pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unexpected context pack schema")
        items = _decode_items(payload["items"], "items")
        digest = _require_digest(payload["digest"], "digest")
        if digest != _pack_digest(items):
            raise ContextPackError("context pack digest does not match its items")
        byte_count = payload["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise ContextPackError("byte_count must be a non-negative integer")
        if byte_count != len(data):
            raise ContextPackError("byte_count does not match the serialized size")
        return cls(items, digest, byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A prefix reuse delta that turns one pack into the next pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Reuse the longest shared prefix and carry the target suffix."""

        shared = 0
        common = min(len(base.items), len(target.items))
        while shared < common and base.items[shared] == target.items[shared]:
            shared += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=shared,
            appended=target.items[shared:],
        )

    @property
    def byte_count(self) -> int:
        """Size of the serialized delta, protocol overhead included."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta to canonical wire JSON."""

        return _canonical({
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_encode(item) for item in self.appended],
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and verify one serialized delta."""

        payload = _load_object(data, _DELTA_FIELDS, "context pack delta")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unexpected context pack delta schema")
        prefix_items = payload["prefix_items"]
        if not isinstance(prefix_items, int) or isinstance(prefix_items, bool) or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        return cls(
            base_digest=_require_digest(payload["base_digest"], "base_digest"),
            target_digest=_require_digest(payload["target_digest"], "target_digest"),
            prefix_items=prefix_items,
            appended=_decode_items(payload["appended"], "appended"),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from one base pack and a delta."""

    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the base pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack item count")
    items = base.items[: delta.prefix_items] + tuple(delta.appended)
    target = ContextPack.build(items)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")
    return target


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": [_encode(item) for item in items]}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _encode(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _decode_items(value: Any, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field} must be an array of Base64 strings")
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{field} must be an array of Base64 strings")
        try:
            items.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field} contains an invalid Base64 item") from error
    return tuple(items)


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        raise ContextPackError(f"{field} must be a sha256 digest")
    return value


def _load_object(data: bytes, expected: frozenset[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} must be a JSON object")
    if set(payload) != expected:
        raise ContextPackError(f"{label} must contain exactly {sorted(expected)}")
    return payload


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
