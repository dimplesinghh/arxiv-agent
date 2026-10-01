"""Tool registry.

The agent loop reads TOOLS to know what's callable; each entry is
(function, json_schema). Adding a tool: write the module, import here,
add to TOOLS. One source of truth — no scattered registrations.
"""

from arxiv_agent.tools import arxiv_search as _arxiv_search
from arxiv_agent.tools import rag_retrieve as _rag_retrieve

TOOLS = {
    "arxiv_search": (_arxiv_search.arxiv_search, _arxiv_search.SCHEMA),
    "rag_retrieve": (_rag_retrieve.rag_retrieve, _rag_retrieve.SCHEMA),
}


def get_schemas() -> list[dict]:
    return [schema for _fn, schema in TOOLS.values()]


def get_function(name: str):
    entry = TOOLS.get(name)
    return entry[0] if entry else None