import re

import requests

from bs4 import BeautifulSoup

from typing import (
    Optional,
    List,
    Dict,
    Any
)


class WebScraper:
    """
    Fetches and extracts readable text from webpages.

    IMPORTANT:

    SourceProcessor decides:

        "Should we visit this URL?"

    WebScraper decides:

        "Is the webpage usable and does its
         actual content relate to the topic?"
    """

    # ========================================================
    # BLOCKED PAGE PHRASES
    # ========================================================

    BLOCK_PHRASES = (

        "access denied",

        "403 forbidden",

        "404 not found",

        "please verify you are human",

        "security check to continue",

        "enable javascript and cookies",

        "attention required! | cloudflare",

        "page not found",

        "checking your browser",

        "captcha"
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        timeout: int = 10,
        max_chars: int = 10000
    ):

        self.timeout = timeout

        self.max_chars = max_chars

        self.headers = {

            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0.0.0 "
                "Safari/537.36"
            ),

            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "*/*;q=0.8"
            ),

            "Accept-Language":
                "en-US,en;q=0.9"
        }

    # ========================================================
    # TOPIC WORD EXTRACTION
    # ========================================================

    def _get_topic_words(
        self,
        topic: str
    ) -> List[str]:
        """
        Extract meaningful words from the topic.
        """

        stop_words = {

            "what",
            "when",
            "where",
            "which",
            "with",
            "from",
            "that",
            "this",
            "about",

            "latest",
            "recent",
            "today",
            "current",
            "currently",
            "new",

            "news",
            "guide",
            "overview",

            "does",
            "have",
            "been",
            "will",
            "would",
            "could",
            "should",

            "your",
            "their",
            "there",
            "these",
            "those",

            "details",
            "information",

            "research",
            "explain",
            "describe",

            "2024",
            "2025",
            "2026",
            "2027"
        }

        words = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            topic.lower()
        )

        return [
            word
            for word in words
            if word not in stop_words
        ]

    # ========================================================
    # CONTENT RELEVANCE
    # ========================================================

    def _is_relevant(
        self,
        text: str,
        topic: str,
        minimum_ratio: float = 0.40
    ) -> bool:
        """
        Check the ACTUAL webpage content.

        SourceProcessor checks the Tavily title/snippet.

        This method performs a second check after
        the webpage has actually been downloaded.
        """

        if not topic:
            return True

        topic_words = (
            self._get_topic_words(
                topic
            )
        )

        if not topic_words:
            return True

        # Only inspect the first 5000 characters
        # for the relevance decision.
        body_lower = (
            text[:5000]
            .lower()
        )

        matched_words = []

        for word in topic_words:

            pattern = (
                r"\b"
                + re.escape(word)
                + r"\b"
            )

            if re.search(
                pattern,
                body_lower
            ):

                matched_words.append(
                    word
                )

        ratio = (
            len(matched_words)
            / len(topic_words)
        )

        return (
            ratio >= minimum_ratio
        )

    # ========================================================
    # SINGLE URL SCRAPER
    # ========================================================

    def scrape(
        self,
        url: str,
        topic: str = ""
    ) -> Optional[str]:
        """
        Fetch one webpage and return readable text.

        Returns None if:

        - URL is invalid
        - request fails
        - page isn't HTML
        - page is too short
        - page looks like an anti-bot page
        - page content is not relevant
        """

        # ----------------------------------------------------
        # 1. Validate URL
        # ----------------------------------------------------

        if not url:
            return None

        if not url.startswith(
            (
                "http://",
                "https://"
            )
        ):

            return None

        # ----------------------------------------------------
        # 2. Skip binary/document URLs
        # ----------------------------------------------------

        if url.lower().endswith(
            (
                ".pdf",
                ".doc",
                ".docx",
                ".zip",
                ".exe",
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".svg",
                ".webp",
                ".mp4",
                ".mp3",
                ".wav",
                ".avi",
                ".mov"
            )
        ):

            print(
                f"       [REJECT] Binary file: "
                f"{url}"
            )

            return None

        try:

            # ------------------------------------------------
            # 3. Download
            # ------------------------------------------------

            print(
                f"       [SCRAPE] {url}"
            )

            response = requests.get(

                url,

                headers=self.headers,

                timeout=self.timeout,

                allow_redirects=True
            )

            response.raise_for_status()

            # ------------------------------------------------
            # 4. Check content type
            # ------------------------------------------------

            content_type = (
                response.headers
                .get(
                    "Content-Type",
                    ""
                )
                .lower()
            )

            if (
                "text/html"
                not in content_type

                and

                "application/xhtml"
                not in content_type
            ):

                print(
                    "       [REJECT] "
                    "Not HTML"
                )

                return None

            # ------------------------------------------------
            # 5. Parse HTML
            # ------------------------------------------------

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # ------------------------------------------------
            # 6. Remove non-content elements
            # ------------------------------------------------

            for element in soup([

                "script",

                "style",

                "noscript",

                "header",

                "footer",

                "nav",

                "svg",

                "form",

                "aside",

                "iframe",

                "button"
            ]):

                element.extract()

            # ------------------------------------------------
            # 7. Extract readable text
            # ------------------------------------------------

            text = soup.get_text(

                separator=" ",

                strip=True
            )

            # ------------------------------------------------
            # 8. Clean excessive whitespace
            # ------------------------------------------------

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            # ------------------------------------------------
            # 9. Check minimum content
            # ------------------------------------------------

            if not text:

                print(
                    "       [REJECT] "
                    "Empty page"
                )

                return None

            if len(text) < 150:

                print(
                    "       [REJECT] "
                    "Page contains too little text"
                )

                return None

            # ------------------------------------------------
            # 10. Anti-bot/error page check
            # ------------------------------------------------

            intro_text = (
                text[:700]
                .lower()
            )

            for phrase in self.BLOCK_PHRASES:

                if phrase in intro_text:

                    print(
                        f"       [REJECT] "
                        f"Blocked/error page: "
                        f"{phrase}"
                    )

                    return None

            # ------------------------------------------------
            # 11. ACTUAL CONTENT RELEVANCE
            # ------------------------------------------------

            if not self._is_relevant(
                text,
                topic
            ):

                print(
                    "       [REJECT] "
                    "Actual webpage content "
                    "does not sufficiently "
                    "match the topic"
                )

                return None

            # ------------------------------------------------
            # 12. Limit content
            # ------------------------------------------------

            text = text[
                :self.max_chars
            ]

            print(
                f"       [ACCEPT] "
                f"{len(text)} characters"
            )

            return text

        except requests.RequestException as e:

            print(
                f"       [REJECT] "
                f"Request failed: {e}"
            )

            return None

        except Exception as e:

            print(
                f"       [REJECT] "
                f"Scraping error: {e}"
            )

            return None

    # ========================================================
    # SCRAPE FILTERED SOURCES
    # ========================================================

    def scrape_relevant_content(
        self,
        sources: List[Dict[str, Any]],
        topic: str = ""
    ) -> List[Dict[str, Any]]:
        """
        IMPORTANT:

        `sources` must already have passed
        SourceProcessor.

        This function does NOT search for more websites.

        It ONLY scrapes the supplied relevant sources.
        """

        findings = []

        print(
            f"\nScraper received "
            f"{len(sources)} filtered sources."
        )

        for index, source in enumerate(
            sources,
            start=1
        ):

            url = source.get(
                "url",
                ""
            )

            title = source.get(
                "title",
                "Untitled Source"
            )

            print(
                f"\n[{index}/{len(sources)}] "
                f"{title}"
            )

            # ------------------------------------------------
            # Scrape
            # ------------------------------------------------

            content = self.scrape(
                url,
                topic=topic
            )

            # ------------------------------------------------
            # Failed
            # ------------------------------------------------

            if not content:

                continue

            # ------------------------------------------------
            # Successful finding
            # ------------------------------------------------

            findings.append({

                "title":
                    title,

                "url":
                    url,

                "content":
                    content,

                "score":
                    source.get(
                        "score",
                        0.0
                    ),

                "published_date":
                    source.get(
                        "published_date"
                    ),

                "priority":
                    source.get(
                        "priority",
                        "MEDIUM"
                    ),

                "source_query":
                    source.get(
                        "source_query",
                        ""
                    )
            })

        print(
            f"\nScraping completed."
        )

        print(
            f"Useful findings: "
            f"{len(findings)}"
        )

        return findings


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    scraper = WebScraper()

    test_url = (
        "https://example.com"
    )

    content = scraper.scrape(
        test_url,
        topic="example"
    )

    if content:

        print(
            "\nScraping successful."
        )

        print(
            content[:1000]
        )

    else:

        print(
            "\nNo usable content."
        )