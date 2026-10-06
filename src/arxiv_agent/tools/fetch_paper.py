"""fetch_paper tool — download and parse a single arxiv paper by ID.

Idempotent and cached. Second call for the same arxiv_id reads from disk,
no network. Parses PDFs with pymupdf, applies the two-column block sorting
Project 1 established (sort by (y, x) position — unsorted blocks interleave
columns and produce gibberish).

Returns sections as a list of {heading, text} dicts. Heading detection is
heuristic: pymupdf doesn't expose structural metadata, so sections are
inferred from font-size spikes and bold short lines. This is good enough
for arxiv papers (consistent LaTeX styles) and bad for everything else;
that's an accepted limitation documented in the tool description.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

import arxiv
import fitz  # pymupdf

import shutil
from urllib.request import Request, urlopen

from arxiv_agent.schema import ToolError, ToolErrorKind, ToolResult


_CACHE_DIR = Path("evals/cache/pdfs")
_TIMEOUT_SECONDS = 60   # downloads can be slow; arxiv PDFs 1-10 MB

# Arxiv IDs come in two forms:
#   old: cs.LG/0601001           (pre-2007)
#   new: 2401.12345v1 or 2401.12345
# Validate strictly so we fail fast on malformed input from the LLM.
_ARXIV_ID_RE = re.compile(r"^(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)$")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_paper",
        "description": (
            "Download and parse the full text of an arXiv paper by its ID "
            "(e.g. '2401.12345v1' or '2401.12345'). Returns sections with "
            "headings and body text. Cached — repeat calls for the same ID "
            "are free. Use when retrieved chunks aren't enough context and "
            "you need surrounding sections. Slow on first call (5-30s)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv paper ID, e.g. '2401.12345v1'. Version suffix optional.",
                },
            },
            "required": ["arxiv_id"],
        },
    },
}


def _cache_path(arxiv_id: str) -> Path:
    # Normalize: slashes in old-style IDs would break paths. Replace with _.
    safe = arxiv_id.replace("/", "_")
    return _CACHE_DIR / f"{safe}.pdf"


def _download(arxiv_id: str, dest: Path) -> None:
    """Download the PDF. arxiv library handles metadata lookup and rate limits
    (3s). As of arxiv>=4.0 the library no longer downloads PDFs itself, so we
    fetch pdf_url with urllib. num_retries=0 so failures.py owns retry logic.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(delay_seconds=3, num_retries=0)
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise LookupError(f"arxiv_id not found: {arxiv_id}")
    pdf_url = results[0].pdf_url
    if not pdf_url:
        raise LookupError(f"no pdf_url for {arxiv_id}")
    # Explicit User-Agent — arxiv rejects some default urllib UAs with 403.
    req = Request(pdf_url, headers={"User-Agent": "arxiv-agent/0.1 (research tool)"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(req, timeout=30) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    # Atomic-ish rename so a partial download never looks like a complete cache file.
    tmp.rename(dest)

def _parse_sections(pdf_path: Path) -> list[dict[str, str]]:
    """Extract sections from a PDF.

    Heuristic: a line is a heading if it is short (< 80 chars), ends without
    a period, and has a larger font than surrounding body text. The first
    heading starts the first section; everything before it is prepended
    (title, abstract) to section index 0 as a synthetic 'preamble'.
    """
    doc = fitz.open(pdf_path)
    try:
        # Pass 1: collect all text lines with their font sizes and bbox.
        # Blocks are sorted by (y, x) per Project 1's two-column fix.
        lines: list[dict] = []
        body_font_sizes: list[float] = []

        for page in doc:
            page_dict = page.get_text("dict", sort=True)  # sort=True does the (y, x) sort
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    size = max(s["size"] for s in spans)
                    lines.append({"text": text, "size": size})
                    body_font_sizes.append(size)

        if not lines:
            return []

        # Body font = median. Headings are above this by a visible margin.
        body_font_sizes.sort()
        body_font = body_font_sizes[len(body_font_sizes) // 2]
        heading_threshold = body_font + 0.5

        # Pass 2: fold lines into sections.
        sections: list[dict[str, str]] = [{"heading": "preamble", "text": ""}]
        for line in lines:
            is_heading = (
                line["size"] >= heading_threshold
                and len(line["text"]) < 80
                and not line["text"].rstrip().endswith((".", "?", "!"))
            )
            if is_heading:
                sections.append({"heading": line["text"], "text": ""})
            else:
                sections[-1]["text"] += line["text"] + " "

        # Drop empty sections and trim whitespace.
        return [
            {"heading": s["heading"], "text": s["text"].strip()}
            for s in sections
            if s["text"].strip()
        ]
    finally:
        doc.close()


def _do_fetch(arxiv_id: str) -> dict[str, Any]:
    """Fetch (from cache or network) and parse. Runs inside the timeout wrapper."""
    cache_file = _cache_path(arxiv_id)
    cache_hit = cache_file.exists()

    if not cache_hit:
        _download(arxiv_id, cache_file)

    sections = _parse_sections(cache_file)
    return {
        "arxiv_id": arxiv_id,
        "sections": sections,
        "num_sections": len(sections),
        "cache_hit": cache_hit,
    }


def fetch_paper(arxiv_id: str) -> ToolResult:
    """Pure function. Returns ToolResult. Never raises into the loop."""
    t0 = time.perf_counter()

    # Validate shape before touching network — garbage IDs fail fast.
    if not _ARXIV_ID_RE.match(arxiv_id):
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message=f"malformed arxiv_id: {arxiv_id!r}",
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_fetch, arxiv_id)
            try:
                data = future.result(timeout=_TIMEOUT_SECONDS)
            except FuturesTimeout:
                return ToolResult(
                    ok=False,
                    error=ToolError(
                        kind=ToolErrorKind.TOOL_TIMEOUT,
                        message=f"fetch_paper exceeded {_TIMEOUT_SECONDS}s",
                        retryable=True,
                    ),
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
    except LookupError as e:
        # arxiv_id doesn't exist. Not retryable — the ID is wrong.
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message=str(e),
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_RETRYABLE,
                message=f"{type(e).__name__}: {e}",
                retryable=True,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    if not data["sections"]:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.EMPTY_RESULT,
                message=f"no parseable sections in {arxiv_id}",
                retryable=False,
            ),
            latency_ms=latency_ms,
        )

    return ToolResult(ok=True, data=data, latency_ms=latency_ms)