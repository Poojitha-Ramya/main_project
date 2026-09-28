import os
import chromadb
from typing import List, Dict, Any, Optional, Union


class VectorStore:
    """
    Vector database powered by ChromaDB.
    Provides fast, indexed cosine similarity search using HNSW for document chunks.
    """

    DEFAULT_COLLECTION_NAME = "research_chunks"
    DEFAULT_PERSIST_DIRECTORY = "./chroma_db"

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = DEFAULT_PERSIST_DIRECTORY
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Optional disk directory for persistent storage.
                               If None, operates in-memory for fast ephemeral research.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._collection = None
        self._init_client()

    def _init_client(self):
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.Client()

    def _get_collection(self):
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self._init_client()
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    @property
    def collection(self):
        if self._collection is not None:
            try:
                # Fast probe to check if collection reference is still valid in ChromaDB
                _ = self._collection.count()
                return self._collection
            except Exception:
                pass
        self._collection = self._get_collection()
        return self._collection

    def _sanitize_metadata(
        self,
        meta: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        ChromaDB metadata values must be primitive types (str, int, float, bool)
        and cannot be None or nested dicts.
        """
        clean: Dict[str, Union[str, int, float, bool]] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Adds or upserts text chunks and their vector embeddings into ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries containing 'chunk_id', 'text', and 'metadata'.
            embeddings: Corresponding vector embeddings from EmbeddingEngine.
            
        Returns:
            Total count of documents currently stored in the collection.
        """
        if not chunks or not embeddings:
            return self.size()

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})."
            )

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        seen_ids = set()

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Chunk at index {idx} is missing chunk_id.")
            chunk_id = str(chunk_id)

            # Guard: ensure every ID in the batch is strictly unique for ChromaDB upsert
            if chunk_id in seen_ids:
                chunk_id = f"{chunk_id}_dup{idx}"
            seen_ids.add(chunk_id)

            text = str(chunk.get("text", "")).strip()
            if not text:
                continue

            raw_meta = chunk.get("metadata", {})
            clean_meta = self._sanitize_metadata(raw_meta)

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(clean_meta)
            embeddings_list.append(list(emb))

        if ids:
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings_list,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"[VectorStore] Upsert recovered from error: {e}")
                self._collection = self._get_collection()
                try:
                    self._collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings_list,
                        metadatas=metadatas
                    )
                except Exception as e2:
                    if "dimension" in str(e2).lower():
                        print(f"[VectorStore] Dimension mismatch detected ({e2}). Re-initializing collection...")
                        try:
                            self.client.delete_collection(name=self.collection_name)
                        except Exception:
                            pass
                        self._collection = self._get_collection()
                        self._collection.upsert(
                            ids=ids,
                            documents=documents,
                            embeddings=embeddings_list,
                            metadatas=metadatas
                        )
                    else:
                        raise

        return self.size()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Performs vector cosine similarity search in ChromaDB to retrieve top-k chunks.
        
        Args:
            query_embedding: Search vector.
            top_k: Number of most similar chunks to retrieve.
            min_score: Minimum cosine similarity threshold (0.0 to 1.0).
            
        Returns:
            List of matching chunk dicts with 'chunk_id', 'text', 'metadata', and 'similarity'.
        """
        total_count = self.size()
        if total_count == 0 or not query_embedding:
            return []

        # Chroma query requires n_results <= total count
        n_results = min(top_k, total_count)

        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        results: List[Dict[str, Any]] = []

        ids_list = response.get("ids", [[]])[0]
        docs_list = response.get("documents", [[]])[0]
        metas_list = response.get("metadatas", [[]])[0]
        dists_list = response.get("distances", [[]])[0]

        for chunk_id, doc, meta, distance in zip(ids_list, docs_list, metas_list, dists_list):
            # With hnsw:space = "cosine", distance is (1 - cosine_similarity).
            # Therefore, similarity = 1.0 - distance.
            similarity = round(max(0.0, 1.0 - float(distance)), 4)

            if similarity >= min_score:
                results.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta or {},
                    "similarity": similarity
                })

        return results

    def clear(self) -> None:
        """
        Resets and clears the collection in ChromaDB.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def size(self) -> int:
        """
        Returns the total number of document chunks stored in ChromaDB.
        """
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics of the ChromaDB collection.
        """
        return {
            "backend": "ChromaDB",
            "collection_name": self.collection_name,
            "count": self.size(),
            "persistent": bool(self.persist_directory),
            "persist_directory": self.persist_directory
        }


if __name__ == "__main__":
    store = VectorStore(
        collection_name="demo_chroma",
        persist_directory=None
    )
    mock_chunks = [
        {"chunk_id": "c1", "text": "Solid-state batteries achieve 450 Wh/kg.", "metadata": {"source": "battery.com", "valid": True}},
        {"chunk_id": "c2", "text": "Quantum computers operate with superconducting qubits.", "metadata": {"source": "quantum.com", "qubits": 1000}},
    ]
    # 3-dim mock embeddings
    mock_embeddings = [
        [0.9, 0.1, 0.0],
        [0.0, 0.1, 0.9]
    ]
    store.add_documents(mock_chunks, mock_embeddings)
    print("ChromaDB count:", store.size())

    res = store.search([0.85, 0.15, 0.0], top_k=1)
    print("ChromaDB search result:", res[0]["chunk_id"], "Similarity:", res[0]["similarity"])
