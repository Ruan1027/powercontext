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

"""Page level PDF capture and evidence page retrieval."""

from __future__ import annotations

import hashlib
import io
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import MemoryCitation

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_DEFINITION_VERSION = "1"

# Retrieval keeps the question's own wording but drops the boilerplate that
# FinanceBench appends to force a response format.
_QUESTION_STOPWORDS = frozenset([
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "from",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "i",
    "we",
    "you",
    "it",
    "its",
    "our",
    "your",
    "this",
    "that",
    "these",
    "those",
    "please",
    "there",
    "their",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
    "any",
    "some",
    "all",
    "each",
    "both",
    "more",
    "most",
    "other",
    "such",
    "than",
    "then",
    "also",
    "just",
    "only",
    "very",
    "much",
    "many",
    "if",
    "not",
    "no",
    "but",
    "so",
    "because",
    "about",
    "into",
    "over",
    "under",
    "above",
    "below",
    "between",
    "during",
    "before",
    "after",
    "has",
    "have",
    "had",
    "having",
    "expressed",
    "stated",
    "reported",
    "please",
    "respond",
    "answer",
    "give",
    "using",
    "use",
    "used",
    "shown",
    "shown",
    "details",
    "detail",
    "perspective",
    "analyst",
    "investment",
    "assume",
    "calculated",
    "calculate",
])

_QUARTER = re.compile(r"\bfy\s*((?:19|20)\d{2})\b")
_TOKEN = re.compile(r"[^a-z0-9]+")
_INSTRUCTION_MARKERS = (
    "respond to the question",
    "give a response to the question",
    "answer in ",
    "if the facts",
    "if the evidence",
    "if the information",
    "if restructuring costs are not explicitly",
)


# General finance-report vocabulary. A question often names a topic with a word the
# filing never uses ("legal battles" versus "Item 3. Legal Proceedings"), so the focused
# queries below also carry the register the document itself uses.
_SYNONYMS = {
    "acquisition": "acquisitions acquired",
    "acquisitions": "acquisition acquired",
    "battle": "litigation proceedings claims",
    "battles": "litigation proceedings claims",
    "borrowings": "debt indebtedness",
    "capex": "capital expenditures spending",
    "cash": "liquidity equivalents",
    "competition": "competitive competitors compete",
    "competitors": "competition competitive",
    "cost": "costs expenses charges",
    "costs": "cost expenses charges",
    "customer": "customers clients",
    "customers": "customer clients",
    "debt": "borrowings indebtedness",
    "dividend": "dividends payout",
    "dividends": "dividend payout",
    "employee": "employees workforce associates headcount",
    "employees": "employee workforce associates headcount",
    "expenses": "expense costs charges",
    "geographies": "geographic geographical regions countries territories segments",
    "geography": "geographic regions countries territories",
    "headcount": "employees workforce associates",
    "impairment": "write charges",
    "inventories": "inventory",
    "inventory": "inventories",
    "lawsuit": "litigation proceedings claims",
    "lawsuits": "litigation proceedings claims",
    "lease": "leases rental",
    "leases": "lease rental",
    "legal": "litigation lawsuits proceedings claims regulatory",
    "liquidity": "cash capital resources",
    "litigation": "legal lawsuits proceedings claims",
    "margin": "margins",
    "margins": "margin",
    "operate": "operates operating operations",
    "operates": "operate operating operations",
    "pension": "retirement benefit plans",
    "regions": "regional geographic countries territories",
    "restructuring": "severance impairment productivity charges",
    "revenue": "revenues sales",
    "revenues": "revenue sales",
    "severance": "restructuring termination",
    "supplier": "suppliers vendors",
    "suppliers": "supplier vendors",
    "tax": "taxes taxation effective",
    "workforce": "employees headcount associates",
}


class PDFDocumentInput(BaseModel):
    """One complete PDF file to split into page Sources."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical PDF page with its extractable text."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured PDF page that can be cited as evidence for one question."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Derive the stable page identity from the document and its revision digest."""

    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        payload = f"{document_id}\x00{document_sha256}\x00{page_number}".encode()
        source_id = f"pdf:{hashlib.sha256(payload).hexdigest()}"
    return source_id


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Extract every physical page of one PDF, in order, as page inputs.

    The whole document is parsed before anything is returned, so a damaged file
    never yields a partial page set.
    """

    document_id = value.document_id
    if not isinstance(document_id, str) or not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    document_sha256 = hashlib.sha256(content).hexdigest()

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
    except Exception as error:  # pypdf raises a wide range of reader errors
        raise ValueError(f"cannot parse PDF document: {error}") from error  # noqa: TRY003

    # An owner password with an empty user password is still a readable document.
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF requires a user password")  # noqa: TRY003

    pages: list[PDFPageInput] = []
    try:
        for index, page in enumerate(reader.pages):
            extracted = page.extract_text()
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=index + 1,
                    text=extracted.strip() if isinstance(extracted, str) else "",
                )
            )
    except Exception as error:  # a damaged page must fail the whole document
        raise ValueError(f"cannot parse PDF document: {error}") from error  # noqa: TRY003
    if not pages:
        raise ValueError("PDF document must contain at least one page")  # noqa: TRY003
    return tuple(pages)


async def search_pdf_pages(
    memory: Any,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` exact page citations that support ``question``."""

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import SearchMemoryRequest

    if limit <= 0 or not isinstance(question, str) or not question.strip():
        return ()

    pool = min(max(limit, 20), 50)
    citations: list[MemoryCitation] = []
    seen: set[str] = set()
    # The question's own wording is the most precise query, so it fills the result
    # first. Focused queries only contribute when it returns fewer pages than asked
    # for, which is what happens for long multi-clause questions.
    for query in _query_variants(question):
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=pool))
        for hit in page.hits:
            if hit.entry_id in seen:
                continue
            seen.add(hit.entry_id)
            citations.append(
                MemoryCitation(
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
            )
            if len(citations) >= limit:
                return tuple(citations)
    return tuple(citations)


def _query_variants(question: str) -> tuple[str, ...]:
    """Broad question wording first, then one focused query per distinctive term."""

    keywords = _keyword_query(question)
    variants: list[str] = []
    if keywords:
        variants.append(keywords)
        for term in dict.fromkeys(keywords.split()):
            synonyms = _SYNONYMS.get(term)
            focused = f"{term} {synonyms}" if synonyms else term
            if focused != keywords:
                variants.append(focused)
    stripped = question.strip()
    if stripped:
        variants.append(stripped)
    return tuple(dict.fromkeys(variants))


def _keyword_query(question: str) -> str:
    text = question.lower()
    for marker in _INSTRUCTION_MARKERS:
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    # "FY2022" appears as "2022" or "fiscal 2022" inside a filing.
    text = _QUARTER.sub(r"\1", text)
    text = _TOKEN.sub(" ", text)
    terms: list[str] = []
    seen: set[str] = set()
    for term in text.split():
        if len(term) < 2 or term in _QUESTION_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return " ".join(terms)


class PDFPageSourceAdapter:
    """Resolve and read the captured PDF page Source."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            definition_version=PDF_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one PDF page as the canonical text evidence payload."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        metadata: dict[str, JsonValue] = {
            "document_id": source.document_id,
            "document_sha256": source.document_sha256,
            "page_number": source.page_number,
        }
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata=metadata,
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_DEFINITION_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)

__all__ = [
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
