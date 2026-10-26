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

"""Row level CSV capture: one Source per record, identified by document and primary key."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_DEFINITION_VERSION = "1"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document to split into row Sources."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One CSV record, kept exactly as it appeared in the document."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured CSV record that can be cited as evidence for a single row."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Derive the stable row identity from its document and primary key value."""

    _require_trimmed_non_empty(document_id, "document_id")
    if not isinstance(primary_key_value, str):
        raise ValueError("primary_key_value must be a string")  # noqa: TRY003, TRY004
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"row source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate one offline CSV document into row inputs.

    The whole document is validated before anything is returned, so a malformed
    document never yields a partial row set.
    """

    _require_trimmed_non_empty(value.document_id, "document_id")
    if not value.primary_key:
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003

    # Accepts a leading UTF-8 BOM, decodes strictly and rejects invalid bytes.
    text = value.content.decode("utf-8-sig")

    # newline="" keeps embedded CRLF inside quoted fields byte faithful.
    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, strict=True)

    header = _read_header(reader)
    key_index = _locate_primary_key(header, value.primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    try:
        for record in reader:
            # A completely empty physical record carries no table data.
            if not record:
                continue
            if len(record) != len(header):
                raise ValueError(  # noqa: TRY003
                    f"record must contain {len(header)} fields, found {len(record)}"
                )
            values = tuple(record)
            primary_key_value = values[key_index]
            if not primary_key_value.strip():
                raise ValueError("primary_key_value must not be blank")  # noqa: TRY003
            if primary_key_value in seen_keys:
                raise ValueError(f"duplicate primary_key_value {primary_key_value!r}")  # noqa: TRY003
            seen_keys.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=primary_key_value,
                    columns=tuple(header),
                    values=values,
                )
            )
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error  # noqa: TRY003
    return tuple(rows)


def _read_header(reader: csv.reader) -> list[str]:
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV document must contain a header record") from None  # noqa: TRY003
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error  # noqa: TRY003
    if not header:
        raise ValueError("CSV document must contain a header record")  # noqa: TRY003
    seen: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("header fields must not be blank")  # noqa: TRY003
        if column in seen:
            raise ValueError(f"duplicate header field {column!r}")  # noqa: TRY003
        seen.add(column)
    return header


def _locate_primary_key(header: list[str], primary_key: str, /) -> int:
    # Header names are compared exactly: surrounding spaces are part of the name.
    for index, column in enumerate(header):
        if column == primary_key:
            return index
    raise ValueError(f"primary_key {primary_key!r} is not a header field")  # noqa: TRY003


def _require_trimmed_non_empty(value: str, field: str, /) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without surrounding whitespace")  # noqa: TRY003


class CSVRowSourceAdapter:
    """Resolve and read the row level CSV Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            definition_version=CSV_ROW_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one CSV row as the canonical text evidence payload."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_DEFINITION_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
