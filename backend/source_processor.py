import re
import sys

# Ensure stdout prints arbitrary unicode characters without crashing on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from datetime import datetime

from typing import List, Dict, Any

from urllib.parse import (
    urlparse,
    parse_qsl,
    urlencode,
    urlunparse
)


class SourceProcessor:
    """
    Filters Tavily search results BEFORE scraping.

    Pipeline:

        Tavily results
             ↓
        URL validation
             ↓
        Domain filtering
             ↓
        File filtering
             ↓
        Relevance score
             ↓
        Topic relevance
             ↓
        Spam filtering
             ↓
        Duplicate removal
             ↓
        Recency ranking
             ↓
        Maximum source limit
             ↓
        WebScraper
    """

    # ========================================================
    # BLOCKED DOMAINS
    # ========================================================

    BLOCKED_DOMAINS = {

        # Social media
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "tiktok.com",
        "youtube.com",
        "youtu.be",
        "pinterest.com",

        # General education aggregators.
        #
        # We block these for the first version because
        # they were causing irrelevant college results.
        "shiksha.com",
        "collegedunia.com",
        "careers360.com",
        "getmyuni.com",
        "targetstudy.com",

        # Generic job/result aggregation sites
        "sarkariresult.com",
        "freshersworld.com",
        "jagranjosh.com",
    }

    # ========================================================
    # IGNORED FILE EXTENSIONS
    # ========================================================

    IGNORED_EXTENSIONS = (

        ".pdf",
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
        ".mov",

        ".zip",
        ".tar",
        ".gz",
        ".exe",
        ".bin",

        ".doc",
        ".docx",
        ".ppt",
        ".pptx"
    )

    # ========================================================
    # TRACKING PARAMETERS
    # ========================================================

    TRACKING_PARAMS = {

        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",

        "fbclid",
        "gclid",

        "ref",
        "source"
    }

    # ========================================================
    # GENERIC SPAM PHRASES
    #
    # IMPORTANT:
    #
    # Do NOT put "faculty details", "fees", "cutoff",
    # "admission", etc. here.
    #
    # Those can be legitimate research topics.
    # ========================================================

    SPAM_TITLE_KEYWORDS = {

        "job fair",

        "seat allotment",

        "hall ticket",

        "answer key",

        "admit card",

        "placement reviews"
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        min_score: float = 0.55
    ):
        """
        min_score:
            Minimum Tavily relevance score.

        0.55 is a reasonable starting point.
        You can tune it later based on testing.
        """

        self.min_score = min_score

    # ========================================================
    # URL NORMALIZATION
    # ========================================================

    def normalize_url(
        self,
        url: str
    ) -> str:
        """
        Normalize a URL so that tracking parameters
        and superficial differences don't create duplicates.
        """

        try:

            parsed = urlparse(
                url.strip()
            )

            scheme = (
                parsed.scheme
                .lower()
            )

            netloc = (
                parsed.netloc
                .lower()
            )

            # Remove www.
            if netloc.startswith(
                "www."
            ):
                netloc = netloc[4:]

            # Remove trailing /
            path = (
                parsed.path
                .rstrip("/")
            )

            # --------------------------------------------
            # Remove tracking parameters
            # --------------------------------------------

            filtered_query = []

            if parsed.query:

                for key, value in parse_qsl(
                    parsed.query,
                    keep_blank_values=False
                ):

                    if (
                        key.lower()
                        not in self.TRACKING_PARAMS
                    ):

                        filtered_query.append(
                            (key, value)
                        )

            query = urlencode(
                filtered_query
            )

            normalized = urlunparse(
                (
                    scheme,
                    netloc,
                    path,
                    "",
                    query,
                    ""
                )
            )

            return normalized

        except Exception:

            return (
                url.strip()
                .lower()
                .rstrip("/")
            )

    # ========================================================
    # DOMAIN CHECK
    # ========================================================

    def is_blocked_domain(
        self,
        domain: str
    ) -> bool:
        """
        Returns True if the domain itself or one of its
        parent domains is blocked.
        """

        domain = (
            domain
            .lower()
            .strip()
        )

        if domain.startswith(
            "www."
        ):
            domain = domain[4:]

        return any(

            domain == blocked

            or domain.endswith(
                "." + blocked
            )

            for blocked
            in self.BLOCKED_DOMAINS
        )

    # ========================================================
    # FILE CHECK
    # ========================================================

    def is_ignored_file(
        self,
        url: str
    ) -> bool:
        """
        Don't send binary/document URLs to the
        normal HTML scraper.
        """

        try:

            parsed = urlparse(
                url
            )

            path = (
                parsed.path
                .lower()
            )

            return path.endswith(
                self.IGNORED_EXTENSIONS
            )

        except Exception:

            return True

    # ========================================================
    # TOPIC WORD EXTRACTION
    # ========================================================

    def get_topic_words(
        self,
        topic: str
    ) -> List[str]:
        """
        Extract meaningful words from the user's topic.
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
            "newest",

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

            "explain",
            "describe",
            "research",

            "details",
            "information",

            "using",
            "used",

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
    # TOPIC RELEVANCE
    # ========================================================

    def is_relevant_to_topic(
        self,
        source: Dict[str, Any],
        topic: str
    ) -> bool:
        """
        Check whether title/snippet contains enough
        important words from the user's topic.

        IMPORTANT:

        We don't use:

            any(word in text)

        because one matching word is too weak.

        Instead we calculate a match ratio.
        """

        if not topic:
            return True

        topic_words = (
            self.get_topic_words(
                topic
            )
        )

        if not topic_words:
            return True

        title = (
            source.get(
                "title",
                ""
            )
            .lower()
        )

        snippet = (
            source.get(
                "snippet",
                ""
            )
            .lower()
        )

        combined = (
            f"{title} {snippet}"
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
                combined
            ):

                matched_words.append(
                    word
                )

        match_ratio = (
            len(matched_words)
            / len(topic_words)
        )

        # ----------------------------------------------------
        # Example:
        #
        # Topic:
        #   "GMRIT faculty details"
        #
        # Important word:
        #   gmrit
        #   faculty
        #
        # A result should normally contain at least
        # 50% of those important words.
        # ----------------------------------------------------

        if len(topic_words) >= 2:

            return (
                match_ratio >= 0.50
            )

        return (
            match_ratio >= 1.0
        )

    # ========================================================
    # SPAM CHECK
    # ========================================================

    def is_spam_content(
        self,
        source: Dict[str, Any],
        topic: str = ""
    ) -> bool:
        """
        Detect obvious SEO/clickbait content.

        We DO NOT block legitimate topic-specific terms
        such as faculty, fees, cutoff, admission, etc.
        """

        title = (
            source.get(
                "title",
                ""
            )
            .lower()
        )

        snippet = (
            source.get(
                "snippet",
                ""
            )
            .lower()
        )

        combined = (
            f"{title} {snippet}"
        )

        topic_lower = (
            topic.lower()
            if topic
            else ""
        )

        # ----------------------------------------------------
        # If the user specifically asks about these topics,
        # don't call them spam.
        # ----------------------------------------------------

        allowed_intents = {

            "admission",
            "cutoff",
            "exam",
            "job",
            "recruitment",
            "fees",
            "counselling",
            "placement",
            "faculty"
        }

        if any(
            intent in topic_lower
            for intent in allowed_intents
        ):

            return False

        # ----------------------------------------------------
        # Generic spam phrases
        # ----------------------------------------------------

        return any(
            spam_phrase in combined
            for spam_phrase
            in self.SPAM_TITLE_KEYWORDS
        )

    # ========================================================
    # RECENCY
    # ========================================================

    def get_recency_priority(
        self,
        source: Dict[str, Any]
    ) -> str:
        """
        Assign:

            HIGH
            MEDIUM
            LOW

        based on available date information.

        Unknown date = MEDIUM.

        We don't automatically reject unknown dates.
        """

        now = datetime.now()

        published_date = source.get(
            "published_date"
        )

        # ----------------------------------------------------
        # Tavily publication date
        # ----------------------------------------------------

        if published_date:

            try:

                date_string = (
                    str(
                        published_date
                    )
                    .replace(
                        "Z",
                        "+00:00"
                    )
                )

                article_date = (
                    datetime.fromisoformat(
                        date_string
                    )
                )

                if article_date.tzinfo:

                    article_date = (
                        article_date
                        .replace(
                            tzinfo=None
                        )
                    )

                age_days = (
                    now - article_date
                ).days

                if age_days <= 7:
                    return "HIGH"

                if age_days <= 30:
                    return "MEDIUM"

                return "LOW"

            except Exception:
                pass

        # ----------------------------------------------------
        # Check relative date in title/snippet
        # ----------------------------------------------------

        combined = (
            f"{source.get('title', '')} "
            f"{source.get('snippet', '')}"
        )

        if re.search(
            r"\b([1-7])\s+"
            r"(day|hour|minute)s?\s+ago\b",
            combined,
            re.IGNORECASE
        ):

            return "HIGH"

        if re.search(
            r"\b\d+\s+"
            r"(week|month|year)s?\s+ago\b",
            combined,
            re.IGNORECASE
        ):

            return "LOW"

        # ----------------------------------------------------
        # Unknown date
        # ----------------------------------------------------

        return "MEDIUM"

    # ========================================================
    # FILTER SOURCES
    # ========================================================

    def filter_sources(
        self,
        sources: List[Dict[str, Any]],
        topic: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Apply all strict filters.

        IMPORTANT:

        There is NO fallback here.

        If a source fails the relevance filter,
        it stays rejected.

        This prevents irrelevant websites from
        coming back just because we need more results.
        """

        filtered = []

        for source in sources:

            url = (
                source.get(
                    "url",
                    ""
                )
                .strip()
            )

            title = (
                source.get(
                    "title",
                    ""
                )
                .strip()
            )

            # ------------------------------------------------
            # 1. Basic validation
            # ------------------------------------------------

            if not url:
                continue

            if not title:
                continue

            parsed = urlparse(
                url
            )

            if parsed.scheme.lower() not in {
                "http",
                "https"
            }:

                continue

            # ------------------------------------------------
            # 2. Domain filtering
            # ------------------------------------------------

            if self.is_blocked_domain(
                parsed.netloc
            ):

                print(
                    f"  [REJECT] Blocked domain: "
                    f"{url}"
                )

                continue

            # ------------------------------------------------
            # 3. File filtering
            # ------------------------------------------------

            if self.is_ignored_file(
                url
            ):

                print(
                    f"  [REJECT] File type: "
                    f"{url}"
                )

                continue

            # ------------------------------------------------
            # 4. Tavily relevance score
            # ------------------------------------------------

            try:

                score = float(
                    source.get(
                        "score",
                        0.0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                score = 0.0

            if score < self.min_score:

                print(
                    f"  [REJECT] Low score "
                    f"{score:.2f}: "
                    f"{title}"
                )

                continue

            # ------------------------------------------------
            # 5. Spam check
            # ------------------------------------------------

            if self.is_spam_content(
                source,
                topic=topic
            ):

                print(
                    f"  [REJECT] Spam/SEO: "
                    f"{title}"
                )

                continue

            # ------------------------------------------------
            # 6. Topic relevance
            # ------------------------------------------------

            if not self.is_relevant_to_topic(
                source,
                topic
            ):

                print(
                    f"  [REJECT] Topic mismatch: "
                    f"{title}"
                )

                continue

            # ------------------------------------------------
            # ACCEPT
            # ------------------------------------------------

            print(
                f"  [ACCEPT] "
                f"{score:.2f} | "
                f"{title}"
            )

            filtered.append(
                source
            )

        return filtered

    # ========================================================
    # DUPLICATE URL REMOVAL
    # ========================================================

    def remove_duplicate_urls(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate URLs after normalization.
        """

        unique_sources = []

        seen_urls = set()

        for source in sources:

            url = (
                source.get(
                    "url",
                    ""
                )
                .strip()
            )

            if not url:
                continue

            normalized = (
                self.normalize_url(
                    url
                )
            )

            if normalized in seen_urls:

                print(
                    f"  [REJECT] Duplicate: "
                    f"{url}"
                )

                continue

            seen_urls.add(
                normalized
            )

            source[
                "normalized_url"
            ] = normalized

            unique_sources.append(
                source
            )

        return unique_sources

    # ========================================================
    # FULL PROCESSING
    # ========================================================

    def process_sources(
        self,
        sources: List[Dict[str, Any]],
        topic: str = "",
        prioritize_recent: bool = False,
        max_sources: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Complete source processing pipeline.

        IMPORTANT ORDER:

        Raw results
            ↓
        Filter
            ↓
        Deduplicate
            ↓
        Recency
            ↓
        Sort
            ↓
        Limit
        """

        print(
            "\n--------------------------------------"
        )

        print(
            "SOURCE PROCESSOR"
        )

        print(
            "--------------------------------------"
        )

        print(
            f"Raw sources: {len(sources)}"
        )

        # ----------------------------------------------------
        # STEP 1 — Strict filtering
        # ----------------------------------------------------

        filtered = (
            self.filter_sources(
                sources,
                topic=topic
            )
        )

        print(
            f"\nAfter filtering: "
            f"{len(filtered)}"
        )

        # ----------------------------------------------------
        # STEP 2 — Deduplicate
        # ----------------------------------------------------

        unique = (
            self.remove_duplicate_urls(
                filtered
            )
        )

        print(
            f"After deduplication: "
            f"{len(unique)}"
        )

        # ----------------------------------------------------
        # STEP 3 — Recency priority
        # ----------------------------------------------------

        priority_weights = {

            "HIGH": 3,

            "MEDIUM": 2,

            "LOW": 1
        }

        for source in unique:

            source[
                "priority"
            ] = self.get_recency_priority(
                source
            )

        # ----------------------------------------------------
        # STEP 4 — Sort
        # ----------------------------------------------------

        if prioritize_recent:

            unique.sort(

                key=lambda source: (

                    priority_weights.get(
                        source.get(
                            "priority",
                            "MEDIUM"
                        ),
                        2
                    ),

                    source.get(
                        "score",
                        0.0
                    )
                ),

                reverse=True
            )

        else:

            unique.sort(

                key=lambda source:
                    source.get(
                        "score",
                        0.0
                    ),

                reverse=True
            )

        # ----------------------------------------------------
        # STEP 5 — Limit
        # ----------------------------------------------------

        final_sources = (
            unique[:max_sources]
        )

        print(
            f"Final sources: "
            f"{len(final_sources)}"
        )

        return final_sources

    # ========================================================
    # URL-ONLY HELPER
    # ========================================================

    def get_relevant_urls(
        self,
        sources: List[Dict[str, Any]],
        topic: str = "",
        prioritize_recent: bool = False,
        max_urls: int = 8
    ) -> List[str]:

        processed = (
            self.process_sources(
                sources,
                topic=topic,
                prioritize_recent=prioritize_recent,
                max_sources=max_urls
            )
        )

        return [
            source.get("url")
            for source in processed
            if source.get("url")
        ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    processor = SourceProcessor(
        min_score=0.55
    )

    sample_sources = [

        {
            "title":
                "GMRIT Faculty Details",

            "url":
                "https://gmrit.edu.in/faculty",

            "snippet":
                "GMR Institute of Technology "
                "faculty and academic staff.",

            "score":
                0.92
        },

        {
            "title":
                "VSM College Faculty Details",

            "url":
                "https://example.com/vsm",

            "snippet":
                "Engineering college faculty "
                "information.",

            "score":
                0.42
        },

        {
            "title":
                "KCET 2026 Counselling",

            "url":
                "https://example.com/kcet",

            "snippet":
                "Karnataka CET counselling.",

            "score":
                0.30
        },

        {
            "title":
                "GMRIT Faculty Details",

            "url":
                "https://gmrit.edu.in/faculty/"
                "?utm_source=google",

            "snippet":
                "GMR Institute of Technology "
                "faculty.",

            "score":
                0.91
        }
    ]

    result = processor.process_sources(
        sample_sources,
        topic="GMRIT faculty details",
        max_sources=8
    )

    print(
        "\nFINAL RELEVANT SOURCES:"
    )

    for source in result:

        print(
            source["title"]
        )

        print(
            source["url"]
        )

        print(
            "Score:",
            source["score"]
        )

        print(
            "Priority:",
            source["priority"]
        )

        print("-" * 60)