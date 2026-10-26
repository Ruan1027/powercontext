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

"""Pi session history capture: raw visible events along the selected branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_DEFINITION_VERSION = "1"
PI_SESSION_FORMAT_VERSION = 3

_EVENT_TYPE = "type"
_CHECKPOINT_VERSION = 1


class PiSessionInput(BaseModel):
    """One complete Pi session archive plus the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi event together with its untouched input record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured Pi event that can be cited as raw evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Derive the stable event identity from its session and event identifiers."""

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")  # noqa: TRY003
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        # Long identifiers cannot be quoted verbatim inside the identity bound.
        digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
        source_id = f"pi:{digest}"
    return source_id


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse one archive and return the visible events from root to the selected leaf.

    The whole archive is validated before anything is returned, so a malformed
    archive never yields a partial event chain.
    """

    text = value.content.decode("utf-8-sig")
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON record: {error}") from error  # noqa: TRY003
        if not isinstance(record, dict):
            raise ValueError("every JSONL record must be a JSON object")  # noqa: TRY003, TRY004
        records.append(record)
    if not records:
        raise ValueError("session archive must not be empty")  # noqa: TRY003

    session_id = _read_session_header(records[0])
    nodes = _index_records(records[1:])
    leaf_id = value.leaf_id if value.leaf_id is not None else _last_node_id(records[1:])
    if leaf_id not in nodes:
        raise ValueError(f"leaf {leaf_id!r} is not present in the archive")  # noqa: TRY003

    events = []
    for event_id in _ancestor_chain(nodes, leaf_id):
        event = _build_event(session_id, nodes[event_id][0])
        if event is not None:
            events.append(event)
    return tuple(events)


def _read_session_header(header: dict[str, Any], /) -> str:
    if header.get(_EVENT_TYPE) != "session":
        raise ValueError("the first record must declare type=session")  # noqa: TRY003
    if header.get("version") != PI_SESSION_FORMAT_VERSION:
        raise ValueError(f"session version must be {PI_SESSION_FORMAT_VERSION}")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session id must be a non-empty string")  # noqa: TRY003
    return session_id


def _index_records(records: list[dict[str, Any]], /) -> dict[str, tuple[dict[str, Any], str | None]]:
    """Map every record id to its payload and parent, rejecting broken graphs."""

    nodes: dict[str, tuple[dict[str, Any], str | None]] = {}
    for record in records:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every record must declare a non-empty string id")  # noqa: TRY003
        if event_id in nodes:
            raise ValueError(f"duplicate record id {event_id!r}")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError(f"record {event_id!r} must declare parentId")  # noqa: TRY003
        parent_id = record["parentId"]
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError(f"record {event_id!r} has an invalid parentId")  # noqa: TRY003
            if parent_id not in nodes:
                raise ValueError(f"record {event_id!r} refers to unknown parent {parent_id!r}")  # noqa: TRY003
        nodes[event_id] = (record, parent_id)
    return nodes


def _last_node_id(records: list[dict[str, Any]], /) -> str:
    if not records:
        raise ValueError("session archive must contain at least one record")  # noqa: TRY003
    return records[-1]["id"]


def _ancestor_chain(nodes: dict[str, tuple[dict[str, Any], str | None]], leaf_id: str, /) -> list[str]:
    chain: list[str] = []
    current: str | None = leaf_id
    while current is not None:
        chain.append(current)
        current = nodes[current][1]
    chain.reverse()
    return chain


def _build_event(session_id: str, record: dict[str, Any], /) -> PiEventInput | None:
    event_type = record.get(_EVENT_TYPE)
    if not isinstance(event_type, str):
        return None

    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        body = _message_body(role, message)
    elif event_type in ("compaction", "branch_summary"):
        role = event_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _content_body(record.get("content"))
    else:
        # Control records keep their graph position but carry no text evidence.
        return None

    if not body.strip():
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=record.get("parentId"),
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, Any], /) -> str:
    if role == "toolResult":
        tool_name = message.get("toolName")
        tool_call_id = message.get("toolCallId")
        text = _content_body(message.get("content"))
        return f"[tool_result {_text_or_unknown(tool_name)} {_text_or_unknown(tool_call_id)}] {text}"
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        return f"$ {command if isinstance(command, str) else ''}\n{output if isinstance(output, str) else ''}"
    return _content_body(message.get("content"))


def _content_body(content: object, /) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get(_EVENT_TYPE)
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            parts.append(f"[tool_call {_text_or_unknown(name)} {_text_or_unknown(call_id)}] {_compact_json(arguments)}")
        # thinking, image and other blocks never become text evidence.
    return "\n".join(parts)


def _compact_json(value: object, /) -> str:
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_or_unknown(value: object, /) -> str:
    return value if isinstance(value, str) and value else "unknown"


def _record_digest(record: dict[str, Any], /) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PiEventSourceAdapter:
    """Resolve and read the captured Pi event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            definition_version=PI_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured Pi event as the canonical text evidence payload."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiSessionConnector:
    """Import the visible events of one Pi branch across restarts and branch switches."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_DEFINITION_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        imported = _load_progress(session.checkpoint)

        digests = {
            pi_event_source_id(event.session_id, event.event_id): _record_digest(event.record) for event in events
        }
        # Detect rewritten history before importing anything, so progress never advances.
        for source_id, digest in digests.items():
            known = imported.get(source_id)
            if known is not None and known != digest:
                raise ValueError(f"captured event {source_id!r} changed after import")  # noqa: TRY003

        progress = dict(imported)
        for event in events:
            source_id = pi_event_source_id(event.session_id, event.event_id)
            if source_id in imported:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            progress[source_id] = digests[source_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": _CHECKPOINT_VERSION, "events": progress},
        )


def _load_progress(checkpoint: JsonValue, /) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}
    return {key: value for key, value in events.items() if isinstance(key, str) and isinstance(value, str)}


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_DEFINITION_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)

__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
