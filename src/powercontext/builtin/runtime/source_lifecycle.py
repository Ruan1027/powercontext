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

"""Propagate Source withdrawal through the evidence DAG of one Scope."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import Column, String, Table, delete, insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.tables import SHARED_METADATA
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_TABLE,
    TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE,
    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE,
)
from powercontext.sources.models import SourceRef

Policy = Literal["all", "any"]
Family = Literal["source", "memory", "topic"]


class SourceLifecycleError(ValueError):
    """A lifecycle registration, event or resolution violated the contract."""


class SourceLifecycleEvent(BaseModel):
    """One immutable availability event for a registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: int
    sequence: int
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("must be a positive integer")  # noqa: TRY003
        return value

    @field_validator("available")
    @classmethod
    def validate_available(cls, value: bool) -> bool:
        if not isinstance(value, bool):
            raise ValueError("must be a boolean")  # noqa: TRY003
        return value


class _Node:
    __slots__ = (
        "available",
        "family",
        "node_id",
        "ordinal",
        "parents",
        "policy",
        "reference",
        "revision",
        "sequence",
    )

    def __init__(self, row: Any) -> None:
        self.node_id = row["node_id"]
        self.family = row["family"]
        self.ordinal = int(row["ordinal"])
        self.reference = json.loads(row["reference"])
        self.parents = tuple(json.loads(row["parents"]))
        self.policy = row["policy"]
        self.revision = int(row["revision"])
        self.sequence = int(row["sequence"])
        self.available = bool(row["available"])

    def supported(self, states: dict[str, bool]) -> bool:
        support = [states.get(parent, False) for parent in self.parents]
        return all(support) if self.policy == "all" else any(support)


class ScopeSourceLifecycle:
    """Register evidence nodes and keep their availability in step with their parents."""

    def __init__(self, runtime: Any, contexts: Any, scope_id: str) -> None:
        self._runtime = runtime
        self._contexts = contexts
        self._scope_id = scope_id
        self._payload_ready = False

    async def register_source(self, node_id: str, reference: SourceRef, *, revision: int = 1) -> None:
        if not isinstance(reference, SourceRef):
            raise SourceLifecycleError("source registration requires a SourceRef")
        await self._register(_node_id(node_id), "source", reference.model_dump(mode="json"), (), "all", revision)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Policy = "all",
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory registration requires a MemoryCitation")
        await self._register(
            _node_id(node_id), "memory", citation.model_dump(mode="json"), parents, _policy(policy), 0
        )

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Policy = "all",
    ) -> None:
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("topic registration requires an ArtifactRef")
        await self._register(_node_id(node_id), "topic", reference.model_dump(mode="json"), parents, _policy(policy), 0)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one availability event and return every currently available node."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires a SourceLifecycleEvent")
        async with self._contexts.database.transaction() as connection:
            rows = await self._load(connection)
            node = rows.get(event.node_id)
            if node is None or node.family != "source":
                raise SourceLifecycleError(f"event node {event.node_id!r} is not a registered Source")
            if node.revision != event.revision:
                raise SourceLifecycleError("event revision does not match the registered Source revision")
            if event.sequence < node.sequence:
                raise SourceLifecycleError("event sequence must not go backwards")
            if event.sequence == node.sequence:
                if node.available != event.available:
                    raise SourceLifecycleError("replayed sequence must not change availability")
                return self._active_ids(rows)
            node.sequence = event.sequence
            node.available = event.available
            await connection.execute(
                update(_table())
                .where(_table().c.scope_id == self._scope_id, _table().c.node_id == node.node_id)
                .values(sequence=event.sequence, available=1 if event.available else 0)
            )
        changed = await self._propagate()
        return changed

    async def resolve(self, node_id: str, *, purpose: str = "generation") -> Any:
        """Return the original reference when the node is usable for that purpose."""

        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError(f"unknown resolve purpose {purpose!r}")
        async with self._contexts.database.transaction() as connection:
            rows = await self._load(connection)
        node = rows.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"node {node_id!r} is not registered")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError(f"node {node_id!r} is not available for generation")
        return _reference(node)

    async def _register(
        self,
        node_id: str,
        family: Family,
        reference: dict[str, Any],
        parents: tuple[str, ...],
        policy: Policy,
        revision: int,
    ) -> None:
        parent_ids = tuple(_node_id(parent) for parent in parents)
        if family != "source":
            if not parent_ids:
                raise SourceLifecycleError("derived nodes require at least one parent")
            if len(set(parent_ids)) != len(parent_ids):
                raise SourceLifecycleError("parents must be unique")
            if node_id in parent_ids:
                raise SourceLifecycleError("a node cannot depend on itself")
        async with self._contexts.database.transaction() as connection:
            rows = await self._load(connection)
            existing = rows.get(node_id)
            if existing is not None:
                if (
                    existing.family == family
                    and existing.reference == reference
                    and existing.parents == parent_ids
                    and existing.policy == policy
                ):
                    return
                raise SourceLifecycleError(f"node {node_id!r} is already bound to another reference")
            for parent in parent_ids:
                if parent not in rows:
                    raise SourceLifecycleError(f"unknown parent {parent!r}")
            ordinal = max((row.ordinal for row in rows.values()), default=0) + 1
            states = {row.node_id: row.available for row in rows.values()}
            if family == "source":
                available = True
            else:
                probe = _Node(_row_view(node_id, family, ordinal, reference, parent_ids, policy, revision, 0, True))
                available = probe.supported(states)
            await connection.execute(
                _insert_statement(
                    self._scope_id, node_id, family, ordinal, reference, parent_ids, policy, revision, 0, available
                )
            )
        if family != "source" and not available:
            await self._set_native_state(node_id, family, reference, available=False)

    async def _propagate(self) -> list[str]:
        transitions: list[tuple[_Node, bool]] = []
        async with self._contexts.database.transaction() as connection:
            nodes = sorted((await self._load(connection)).values(), key=lambda node: node.ordinal)
            states: dict[str, bool] = {}
            for node in nodes:
                if node.family == "source":
                    states[node.node_id] = node.available
                    continue
                supported = node.supported(states)
                if supported != node.available:
                    node.available = supported
                    transitions.append((node, supported))
                states[node.node_id] = node.available
                await connection.execute(
                    update(_table())
                    .where(_table().c.scope_id == self._scope_id, _table().c.node_id == node.node_id)
                    .values(available=1 if node.available else 0)
                )
        for node, available in transitions:
            await self._set_native_state(node.node_id, node.family, node.reference, available=available)
        async with self._contexts.database.transaction() as connection:
            rows = await self._load(connection)
        return self._active_ids(rows)

    @staticmethod
    def _active_ids(rows: dict[str, _Node]) -> list[str]:
        return sorted(node_id for node_id, node in rows.items() if node.available)

    async def _set_native_state(self, node_id: str, family: str, reference: dict[str, Any], *, available: bool) -> None:
        if family == "memory":
            citation = MemoryCitation.model_validate(reference)
            async with self._runtime._context(self._scope_id) as context:
                service = context.artifacts.memory
                # State lives on the head revision; the citation still addresses the old one.
                memory = await service.head(citation.memory_ref.artifact_id)
                entry = next(
                    (item for item in await service.entries(memory) if item.entry_id == citation.entry_id),
                    None,
                )
                if entry is None:
                    raise SourceLifecycleError(f"memory entry for {node_id!r} is not readable")
                if available:
                    await service.reactivate(memory, entries=(entry,), reason="source lifecycle")
                else:
                    await service.forget(memory, entries=(entry,), reason="source lifecycle")
        elif family == "topic":
            ref = ArtifactRef.model_validate(reference)
            async with self._contexts.database.transaction() as connection:
                if available:
                    await self._restore_topic(connection, node_id)
                else:
                    await self._hide_topic(connection, node_id, ref)


    async def _hide_topic(self, connection: Any, node_id: str, ref: ArtifactRef) -> None:
        snapshot: list[list[dict[str, Any]]] = []
        for table in (TOPIC_MEMORY_ACTIVE_TOPICS_TABLE, TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE):
            rows = (
                await connection.execute(
                    select(table).where(table.c.scope_id == self._scope_id, table.c.artifact_id == ref.artifact_id)
                )
            ).mappings().all()
            snapshot.append([dict(row) for row in rows])
            await connection.execute(
                delete(table).where(table.c.scope_id == self._scope_id, table.c.artifact_id == ref.artifact_id)
            )
        await connection.execute(
            delete(_snapshot_table()).where(
                _snapshot_table().c.scope_id == self._scope_id, _snapshot_table().c.node_id == node_id
            )
        )
        await connection.execute(
            insert(_snapshot_table()).values(
                scope_id=self._scope_id, node_id=node_id, payload=json.dumps(snapshot)
            )
        )

    async def _restore_topic(self, connection: Any, node_id: str) -> None:
        stored = (
            await connection.execute(
                select(_snapshot_table().c.payload).where(
                    _snapshot_table().c.scope_id == self._scope_id, _snapshot_table().c.node_id == node_id
                )
            )
        ).scalar()
        if not stored:
            return
        snapshot = json.loads(stored)
        for table, rows in zip((TOPIC_MEMORY_ACTIVE_TOPICS_TABLE, TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE), snapshot):
            for row in rows:
                await connection.execute(insert(table).values(**row))
        await connection.execute(
            delete(_snapshot_table()).where(
                _snapshot_table().c.scope_id == self._scope_id, _snapshot_table().c.node_id == node_id
            )
        )

    async def _load(self, connection: Any) -> dict[str, _Node]:
        if not self._payload_ready:
            await connection.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS pc_source_lifecycle_snapshot ("
                "scope_id VARCHAR(256) NOT NULL, node_id VARCHAR(256) NOT NULL, "
                "payload VARCHAR(200000) NOT NULL, PRIMARY KEY (scope_id, node_id))"
            )
            self._payload_ready = True
        rows = await connection.execute(select(_table()).where(_table().c.scope_id == self._scope_id))
        return {row["node_id"]: _Node(row) for row in rows.mappings().all()}


class SourceLifecycleApplication:
    """Expose the per-Scope lifecycle service from the Runtime."""

    def __init__(self, runtime: Any, contexts: Any) -> None:
        self._runtime = runtime
        self._contexts = contexts

    def for_scope(self, scope_id: str) -> ScopeSourceLifecycle:
        return ScopeSourceLifecycle(self._runtime, self._contexts, scope_id)


def _table() -> Any:
    return SOURCE_LIFECYCLE_TABLE


def _snapshot_table() -> Any:
    return Table(
        "pc_source_lifecycle_snapshot",
        SHARED_METADATA,
        Column("scope_id", String(256), primary_key=True),
        Column("node_id", String(256), primary_key=True),
        Column("payload", String(200000), nullable=False),
        extend_existing=True,
    )


def _node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node id must be a non-empty string")
    return value


def _policy(value: str) -> Policy:
    if value not in ("all", "any"):
        raise SourceLifecycleError("policy must be 'all' or 'any'")
    return value  # type: ignore[return-value]


def _reference(node: _Node) -> Any:
    if node.family == "source":
        return SourceRef.model_validate(node.reference)
    if node.family == "memory":
        return MemoryCitation.model_validate(node.reference)
    return ArtifactRef.model_validate(node.reference)


def _row_view(
    node_id: str,
    family: str,
    ordinal: int,
    reference: dict[str, Any],
    parents: tuple[str, ...],
    policy: str,
    revision: int,
    sequence: int,
    available: bool,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "family": family,
        "ordinal": ordinal,
        "reference": json.dumps(reference),
        "parents": json.dumps(list(parents)),
        "policy": policy,
        "revision": revision,
        "sequence": sequence,
        "available": 1 if available else 0,
    }


def _insert_statement(
    scope_id: str,
    node_id: str,
    family: str,
    ordinal: int,
    reference: dict[str, Any],
    parents: tuple[str, ...],
    policy: str,
    revision: int,
    sequence: int,
    available: bool,
) -> Any:
    from sqlalchemy import insert

    return insert(_table()).values(
        scope_id=scope_id,
        **_row_view(node_id, family, ordinal, reference, parents, policy, revision, sequence, available),
    )


__all__ = [
    "ScopeSourceLifecycle",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
]
