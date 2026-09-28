import re
import hashlib
from typing import List, Dict, Any, Optional


class TextChunker:
    """
    Splits long scraped web articles into semantic, overlapping chunks
    while preserving sentence boundaries and metadata.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Target maximum character length for each chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences based on punctuation and line breaks.
        """
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        # Split on sentence ending punctuation followed by space or newline
        sentences = re.split(r"(?<=[.?!])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunks a single text document into overlapping segments.
        
        Returns:
            List of dictionaries containing chunk_id, text, and metadata.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size and we already have sentences
            if current_length + sentence_len > self.chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences)
                chunks.append(chunk_str)

                # Keep sentences from the end that fit within chunk_overlap
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_sentences = overlap_sentences
                current_length = overlap_len

            current_sentences.append(sentence)
            current_length += sentence_len + 1

        if current_sentences:
            chunk_str = " ".join(current_sentences)
            # Avoid adding duplicate if it was already appended
            if not chunks or chunks[-1] != chunk_str:
                chunks.append(chunk_str)

        # Format chunks with IDs and metadata
        doc_id = metadata.get("url") or metadata.get("title") or "doc"
        doc_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)[:24]
        doc_hash = hashlib.md5(str(doc_id).encode("utf-8")).hexdigest()[:8]

        formatted_chunks = []
        total = len(chunks)
        for idx, content in enumerate(chunks):
            chunk_meta = dict(metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["total_chunks"] = total
            chunk_meta["char_length"] = len(content)

            formatted_chunks.append({
                "chunk_id": f"{doc_slug}_{doc_hash}_{idx}",
                "text": content,
                "metadata": chunk_meta
            })

        return formatted_chunks

    def chunk_findings(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunks multiple findings (from scraper.py) and flattens into a single list of chunks.
        Guarantees all chunk IDs across all findings are unique.
        
        Args:
            findings: List of dicts with 'title', 'url', 'content', etc.
        """
        all_chunks = []
        seen_ids = set()

        for f_idx, finding in enumerate(findings):
            content = finding.get("content", "")
            meta = {
                "title": finding.get("title", "Untitled"),
                "url": finding.get("url", ""),
                "score": finding.get("score", 0.0),
                "published_date": finding.get("published_date"),
                "priority": finding.get("priority", "MEDIUM"),
                "source_query": finding.get("source_query", "")
            }

            chunks = self.chunk_text(content, metadata=meta)
            for c in chunks:
                cid = c["chunk_id"]
                if cid in seen_ids:
                    cid = f"{cid}_{f_idx}"
                seen_ids.add(cid)
                c["chunk_id"] = cid
                all_chunks.append(c)

        return all_chunks


if __name__ == "__main__":
    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    sample_text = (
        "Solid-state batteries replace the liquid electrolyte with a solid one. "
        "This increases energy density and significantly improves safety. "
        "Researchers are testing sulfide and oxide-based solid electrolytes. "
        "Manufacturing at scale remains a key challenge for commercial electric vehicles. "
        "Major automakers expect mass production by 2028."
    )
    res = chunker.chunk_text(sample_text, metadata={"title": "Solid State Overview", "url": "https://example.com/battery"})
    print(f"Generated {len(res)} chunks:")
    for c in res:
        print(f"[{c['chunk_id']}] ({len(c['text'])} chars): {c['text']}")
