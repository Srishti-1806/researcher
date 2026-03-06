# tools/search_tools.py
import os
from typing import Any, List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document

# Lightweight DuckDuckGo wrapper when langchain tools are unavailable
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

# 1. Tavily is the "Gold Standard" for LLM Research Agents
# Note: Requires TAVILY_API_KEY in your .env
def get_tavily_search():
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_raw_content=False, # We want clean snippets for Gemma's context window
        include_answer=True
    )

# 2. DuckDuckGo as a zero-cost fallback
def get_ddg_search():
    # Prefer langchain's wrapper if available, otherwise return a simple wrapper
    if DDGS is None:
        return DuckDuckGoSearchRun()

    class SimpleDDG:
        def invoke(self, query: str) -> List[dict]:
            try:
                with DDGS() as ddg:
                    results = list(ddg.text(query, max_results=5))
                    # Normalize to dict list with url and content keys
                    formatted = []
                    for r in results:
                        formatted.append({
                            "url": r.get("href") or r.get("url") or r.get("source"),
                            "content": r.get("body") or r.get("snippet") or str(r)
                        })
                    return formatted
            except Exception as e:
                return [{"url": "", "content": f"DDG search error: {e}"}]

    return SimpleDDG()

def web_search_tool(query: str, mode: str = "quick"):
    """
    Orchestrates the search based on the execution mode.
    """
    try:
        # Prefer Tavily for deep searches when API key is present
        if mode == "deep" and os.getenv("TAVILY_API_KEY"):
            search = get_tavily_search()
            results = search.invoke({"query": query})
        else:
            search = get_ddg_search()
            results = search.invoke(query)

        # Normalize different result shapes into a readable string
        if isinstance(results, str):
            return results

        if isinstance(results, list):
            formatted = []
            for r in results:
                if isinstance(r, dict):
                    url = r.get("url") or r.get("href") or r.get("source", "")
                    content = r.get("content") or r.get("snippet") or str(r)
                    formatted.append(f"Source: {url}\nContent: {content}")
                else:
                    formatted.append(str(r))
            return "\n\n".join(formatted)

        return str(results)
    except Exception as e:
        return f"Search failed: {str(e)}"


def get_search_tool(mode: str = "quick") -> Any:
    """Return an object with an `invoke` method appropriate for the mode.

    - `deep` will prefer Tavily when available.
    - otherwise returns a DuckDuckGo wrapper.
    """
    if mode == "deep" and os.getenv("TAVILY_API_KEY"):
        return get_tavily_search()
    return get_ddg_search()


__all__ = ["get_tavily_search", "get_ddg_search", "web_search_tool", "get_search_tool"]