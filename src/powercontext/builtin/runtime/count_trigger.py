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

"""Start Memory processing once enough unseen Source entries accumulate."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from sqlalchemy import select

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.tables import SOURCE_JOURNAL_HEADS_TABLE

COUNT_TRIGGER_INTERVAL_SECONDS = 0.2
_COUNT_TRIGGER_PAGE = 64

logger = logging.getLogger(__name__)

FlushScope = Callable[[str], Awaitable[object]]


class SourceCountTrigger:
    """Poll each Family's journal cursor and run one window when the count is reached."""

    def __init__(
        self,
        database: object,
        *,
        families: tuple[tuple[str, FlushScope, int], ...],
        lock: asyncio.Lock | None = None,
        interval: float = COUNT_TRIGGER_INTERVAL_SECONDS,
    ) -> None:
        for name, _flush, threshold in families:
            if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold <= 0:
                raise ValueError(f"{name} threshold must be a positive integer")  # noqa: TRY003
        self._database = database
        self._families = families
        self._lock = lock or asyncio.Lock()
        self._interval = interval
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="source-count-trigger")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            for binding_name, flush, threshold in self._families:
                try:
                    await self._tick(binding_name, flush, threshold)
                except asyncio.CancelledError:
                    raise
                except Exception:  # a failed window keeps its pending entries for the next pass
                    logger.exception("source count trigger pass failed", extra={"binding": binding_name})

    async def _tick(self, binding_name: str, flush: FlushScope, threshold: int) -> None:
        async with self._lock:
            for scope_id in await self._ready_scopes(binding_name, threshold):
                await flush(scope_id)

    async def _ready_scopes(self, binding_name: str, threshold: int) -> tuple[str, ...]:
        heads = SOURCE_JOURNAL_HEADS_TABLE
        statement = select(heads.c.scope_id, heads.c.position).order_by(heads.c.scope_id).limit(_COUNT_TRIGGER_PAGE)
        ready: list[str] = []
        async with self._database.transaction() as connection:  # type: ignore[attr-defined]
            rows = (await connection.execute(statement)).all()
            cursors = SourceCursorRepository()
            for scope_id, position in rows:
                cursor = await cursors.load(connection, scope_id, binding_name)
                seen = 0 if cursor is None else cursor.cursor.sequence
                if int(position) - seen >= threshold:
                    ready.append(str(scope_id))
        return tuple(ready)


__all__ = ["COUNT_TRIGGER_INTERVAL_SECONDS", "SourceCountTrigger"]
