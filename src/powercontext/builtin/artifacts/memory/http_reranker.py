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

"""Rerank Memory candidates with an HTTP cross-encoder scoring service."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference.models import InferenceUsage

HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score coarse candidates over HTTP in batches and select the global best."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate exactly once and order them globally."""

        count = len(candidates)
        if count < 1:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if limit < 1 or limit > count:
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        # Ranks always index the complete pool, so a batch never renumbers them.
        indexed = list(enumerate(candidates, start=1))
        scores: dict[int, float] = {}
        requests = 0
        for offset in range(0, count, self._batch_size):
            batch = indexed[offset : offset + self._batch_size]
            requests += 1
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": hit.text} for rank, hit in batch],
                },
            )
            response.raise_for_status()
            scores.update(_read_batch_scores(response, {rank for rank, _ in batch}))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _read_batch_scores(response: Any, expected_ranks: set[int]) -> dict[int, float]:
    """Validate one batch response and return its scores keyed by global rank."""

    payload = response.json()
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain only scores")  # noqa: TRY003
    entries = payload["scores"]
    if not isinstance(entries, list):
        raise ValueError("rerank scores must be a list")  # noqa: TRY003, TRY004

    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("each score must contain only rank and score")  # noqa: TRY003
        rank = entry["rank"]
        score = entry["score"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("score rank must be an integer")  # noqa: TRY003, TRY004
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score must be a number")  # noqa: TRY003, TRY004
        if not math.isfinite(score):
            raise ValueError("score must be finite")  # noqa: TRY003
        # A missing, repeated or foreign rank would break the global mapping.
        if rank in scores or rank not in expected_ranks:
            raise ValueError(f"score rank {rank!r} does not match this batch")  # noqa: TRY003
        scores[rank] = float(score)
    if set(scores) != expected_ranks:
        raise ValueError("rerank response must cover every batch rank once")  # noqa: TRY003
    return scores


__all__ = ["HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
