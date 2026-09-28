import os

from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from tavily import TavilyClient


# ============================================================
# LOAD .ENV
# ============================================================

ENV_PATH = (
    Path(__file__).resolve().parent.parent / ".env"
)

load_dotenv(dotenv_path=ENV_PATH)


class SearchEngine:
    """
    Handles web searches using Tavily.

    Responsibilities:
    1. Send search queries to Tavily
    2. Support multiple subqueries
    3. Support time filtering
    4. Preserve Tavily relevance scores
    5. Normalize Tavily results into our own format
    """

    # Domains that we do not want Tavily to return.
    #
    # These are broad exclusions. More detailed
    # filtering is performed later by SourceProcessor.
    EXCLUDED_DOMAINS = [
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "tiktok.com",
        "youtube.com",
        "youtu.be",
        "pinterest.com",
    ]

    def __init__(
        self,
        api_key: str = ""
    ):

        self.api_key = (
            api_key
            or os.getenv(
                "TAVILY_API_KEY",
                ""
            )
        )

        if not self.api_key:
            raise ValueError(
                "TAVILY_API_KEY is missing. "
                "Please set it in your .env file."
            )

        self.client = TavilyClient(
            api_key=self.api_key
        )

    # ========================================================
    # SINGLE SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        max_results: int = 5,
        time_range: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute one Tavily search.

        Returns normalized results containing:

        title
        url
        snippet
        score
        published_date
        source_query
        """

        if not query or not query.strip():
            return []

        query = query.strip()

        search_kwargs = {
            "query": query,

            # Advanced gives higher relevance,
            # but costs more API credits.
            "search_depth": "advanced",

            "max_results": max_results,

            # Exclude obvious social sites at the
            # search stage itself.
            "exclude_domains":
                self.EXCLUDED_DOMAINS,
        }

        # ----------------------------------------------------
        # Recent/current research
        # ----------------------------------------------------

        if time_range:
            search_kwargs["time_range"] = time_range

        try:

            response = self.client.search(
                **search_kwargs
            )

        except Exception as e:

            print(
                f"[Tavily] Search failed "
                f"for '{query}': {e}"
            )

            return []

        results = []

        for item in response.get(
            "results",
            []
        ):

            url = item.get(
                "url",
                ""
            )

            title = item.get(
                "title",
                "Untitled Source"
            )

            snippet = item.get(
                "content",
                ""
            )

            score = item.get(
                "score",
                0.0
            )

            # ------------------------------------------------
            # Safely convert score to float
            # ------------------------------------------------

            try:
                score = float(score)
            except (
                TypeError,
                ValueError
            ):
                score = 0.0

            results.append({

                "title": title,

                "url": url,

                "snippet": snippet,

                "score": score,

                # Tavily may return this depending
                # on the search configuration/topic.
                "published_date":
                    item.get(
                        "published_date"
                    ),

                # Very useful because later we can
                # know which subquery generated this URL.
                "source_query": query
            })

        return results

    # ========================================================
    # MULTIPLE SEARCHES
    # ========================================================

    def search_multiple(
        self,
        queries: List[str],
        max_results_per_query: int = 5,
        time_range: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple subqueries sequentially.

        Example:

        queries = [
            "solid state battery breakthroughs",
            "solid state battery energy density",
            "solid state battery commercialization"
        ]

        Every Tavily result is combined into one list.
        """

        all_results = []

        if not queries:
            return []

        print(
            f"\nTavily will execute "
            f"{len(queries)} searches."
        )

        for index, query in enumerate(
            queries,
            start=1
        ):

            print(
                f"\n[Tavily {index}/{len(queries)}]"
            )

            print(
                f"Query: {query}"
            )

            results = self.search(
                query=query,
                max_results=max_results_per_query,
                time_range=time_range
            )

            print(
                f"Results: {len(results)}"
            )

            all_results.extend(
                results
            )

        return all_results


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    engine = SearchEngine()

    results = engine.search(
        "Breakthroughs in solid-state batteries",
        max_results=5
    )

    print(
        f"\nRetrieved {len(results)} results:\n"
    )

    for result in results:

        print(
            f"Title: {result['title']}"
        )

        print(
            f"Score: {result['score']}"
        )

        print(
            f"URL: {result['url']}"
        )

        print(
            f"Query: {result['source_query']}"
        )

        print(
            "-" * 60
        )