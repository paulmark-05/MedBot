# =============================================================
# tools.py — Tool implementations for Medical Health FAQ Bot
# Currently: DuckDuckGo web search for live health information
# =============================================================

from config import WEB_SEARCH_MAX_RESULTS


def web_search(question: str) -> str:
    """
    Searches DuckDuckGo for current health information related to the question.
    Returns a formatted string of results, or an error message.
    Used when the router decides the question needs live/recent information
    (e.g., drug recalls, latest guidelines, recent outbreaks).
    """
    # FIX: Package was renamed from `duckduckgo_search` to `ddgs`.
    # Install: pip install ddgs
    # The old import triggered a RuntimeWarning and will eventually break.
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    f"health medical {question}",
                    max_results=WEB_SEARCH_MAX_RESULTS,
                )
            )

        if not results:
            return "No web search results found for this query."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body  = r.get("body",  "No description")[:300]
            href  = r.get("href",  "")
            formatted.append(f"[{i}] {title}\n{body}\nSource: {href}")

        return "\n\n".join(formatted)

    except ImportError:
        return (
            "Web search unavailable: ddgs not installed. "
            "Run: pip install ddgs"
        )
    except Exception as e:
        return f"Web search error: {str(e)}"