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

"""Coalesce concurrent embedding requests and reuse unchanged vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage

DEFAULT_MAX_BATCH_SIZE = 32


@dataclass(slots=True)
class _Waiter:
    """One caller waiting for its own texts inside a shared provider batch."""

    texts: tuple[str, ...]
    cached: dict[str, tuple[float, ...]]
    missing: tuple[str, ...]
    future: asyncio.Future[EmbeddingResult]


class CoalescingEmbeddingModel:
    """Serve many concurrent embed calls with the fewest bounded provider batches."""

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: int | float = 0,
        cache_size: int | None = None,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if cache_size is not None and (not isinstance(cache_size, int) or cache_size < 0):
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._waiters: list[_Waiter] = []
        self._flusher: asyncio.Task[None] | None = None

    @property
    def profile(self) -> Any:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return vectors in the caller's own order, sharing provider work."""

        requested = tuple(texts)
        if not requested:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        cached: dict[str, tuple[float, ...]] = {}
        missing: list[str] = []
        for text in requested:
            if text in self._cache:
                cached[text] = self._cache[text]
                self._cache.move_to_end(text)
            elif text not in missing:
                missing.append(text)
        if not missing:
            return EmbeddingResult(
                vectors=tuple(cached[text] for text in requested),
                usage=InferenceUsage(requests=0),
            )

        waiter = _Waiter(
            texts=requested,
            cached=cached,
            missing=tuple(missing),
            future=asyncio.get_running_loop().create_future(),
        )
        self._waiters.append(waiter)
        if self._flusher is None:
            self._flusher = asyncio.create_task(self._flush(), name="coalescing-embedding-flush")
        return await waiter.future

    async def _flush(self) -> None:
        try:
            while self._waiters:
                # One short window lets concurrent callers join the same batch.
                await asyncio.sleep(self._coalesce_seconds)
                waiters, self._waiters = self._waiters, []
                await self._drain(waiters)
        finally:
            self._flusher = None
            # A caller that arrived during shutdown still needs its batch served.
            if self._waiters:
                self._flusher = asyncio.create_task(self._flush(), name="coalescing-embedding-flush")

    async def _drain(self, waiters: list[_Waiter]) -> None:  # noqa: C901 - batch fan-out
        live = [waiter for waiter in waiters if not waiter.future.done()]
        if not live:
            # Every caller went away: do not spend a provider request on them.
            return
        pending: list[str] = []
        for waiter in live:
            for text in waiter.missing:
                if text not in pending:
                    pending.append(text)

        vectors: dict[str, tuple[float, ...]] = {}
        batches: list[tuple[set[str], InferenceUsage]] = []
        failed: set[str] = set()
        for start in range(0, len(pending), self._max_batch_size):
            batch = tuple(pending[start : start + self._max_batch_size])
            if set(batch) <= failed:
                continue
            try:
                result = await self._model.embed(batch)
                produced = _batch_vectors(batch, result)
            except Exception as error:  # a failed batch only fails the callers that needed it
                failed.update(batch)
                for waiter in live:
                    if not waiter.future.done() and set(waiter.missing) & set(batch):
                        waiter.future.set_exception(error)
                continue
            for text, vector in produced:
                vectors[text] = vector
                # Cache per successful batch so a later failure never discards it.
                self._remember(text, vector)
            batches.append((set(batch), result.usage))

        for waiter in waiters:
            if waiter.future.done():
                # A cancelled caller never blocks the others.
                continue
            unresolved = [text for text in waiter.missing if text not in vectors]
            if unresolved:
                waiter.future.set_exception(RuntimeError(f"embedding provider did not return {unresolved[0]!r}"))
                continue
            waiter.future.set_result(
                EmbeddingResult(
                    vectors=tuple(vectors[text] if text in vectors else waiter.cached[text] for text in waiter.texts),
                    usage=_usage_for(waiter, batches),
                )
            )

    def _remember(self, text: str, vector: tuple[float, ...]) -> None:
        self._cache[text] = vector
        self._cache.move_to_end(text)
        if self._cache_size is None:
            return
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _batch_vectors(batch: tuple[str, ...], result: EmbeddingResult) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if len(result.vectors) != len(batch):
        raise ValueError("embedding provider returned a mismatched vector count")  # noqa: TRY003
    return tuple(
        (text, tuple(float(value) for value in vector)) for text, vector in zip(batch, result.vectors, strict=True)
    )


def _usage_for(waiter: _Waiter, batches: list[tuple[set[str], InferenceUsage]]) -> InferenceUsage:
    wanted = set(waiter.missing)
    requests = 0
    input_tokens = 0
    output_tokens = 0
    metered = False
    for texts, usage in batches:
        if not wanted & texts:
            continue
        requests += 1
        if usage.input_tokens is not None:
            input_tokens += usage.input_tokens
            metered = True
        if usage.output_tokens is not None:
            output_tokens += usage.output_tokens
            metered = True
    return InferenceUsage(
        requests=requests,
        input_tokens=input_tokens if metered else None,
        output_tokens=output_tokens if metered else None,
    )


__all__ = ["DEFAULT_MAX_BATCH_SIZE", "CoalescingEmbeddingModel"]
