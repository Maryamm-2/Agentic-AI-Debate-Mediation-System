"""Semantic embeddings and FAISS-based retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from typing import TYPE_CHECKING

# Defer heavy imports (faiss, sentence_transformers) to function scope so the module
# can be imported quickly when embeddings are disabled in the config.
if TYPE_CHECKING:
    # Type hints only
    from sentence_transformers import SentenceTransformer  # pragma: no cover

from .chunker import DocumentChunk


@dataclass
class EmbeddingResult:
    """Result from embedding-based retrieval."""
    chunk: DocumentChunk
    similarity: float
    rank: int


class EmbeddingIndex:
    """FAISS-based semantic similarity index."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                # Import here to avoid heavy import at module load time
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError(f"sentence-transformers is required to use embeddings: {e}")

            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build FAISS index from document chunks."""
        self._load_model()

        # Lazy import of faiss since it is heavy
        try:
            import faiss
        except Exception as e:
            raise RuntimeError(f"faiss is required to build embedding index: {e}")

        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        # SentenceTransformer returns a numpy array
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        # Build FAISS index
        dimension = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        print(f"Built FAISS index with {len(chunks)} chunks, dimension {dimension}")
    
    def search(self, query: str, top_k: int = 5) -> List[EmbeddingResult]:
        """Search the index for semantically similar chunks."""
        if not self.index or not self.model:
            return []

        # Lazy import faiss for operations
        try:
            import faiss
        except Exception:
            return []

        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            if idx != -1:  # Valid result
                result = EmbeddingResult(
                    chunk=self.chunks[idx],
                    similarity=float(similarity),
                    rank=rank
                )
                results.append(result)
        
        return results
    
    def save_index(self, path: Path) -> None:
        """Save the embedding index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lazy import faiss
        try:
            import faiss
        except Exception as e:
            raise RuntimeError(f"faiss is required to save embedding index: {e}")

        # Save FAISS index
        faiss_path = path.with_suffix('.faiss')
        faiss.write_index(self.index, str(faiss_path))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'chunks': self.chunks,
            'embeddings': self.embeddings
        }
        
        metadata_path = path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved embedding index to {path}")
    
    def load_index(self, path: Path) -> bool:
        """Load embedding index from disk."""
        try:
            # Lazy import faiss
            import faiss

            # Load FAISS index
            faiss_path = path.with_suffix('.faiss')
            self.index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            metadata_path = path.with_suffix('.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.model_name = metadata['model_name']
            self.chunks = metadata['chunks']
            self.embeddings = metadata['embeddings']
            
            # Load model
            self._load_model()
            
            print(f"Loaded embedding index from {path} with {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"Failed to load embedding index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.index:
            return {}
        
        return {
            'num_chunks': len(self.chunks),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_name': self.model_name,
            'index_type': type(self.index).__name__
        }


class HybridRetriever:
    """Combines BM25 and embedding-based retrieval."""
    
    def __init__(self, bm25_index, embedding_index, bm25_weight: float = 0.7):
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.bm25_weight = bm25_weight
        self.embedding_weight = 1.0 - bm25_weight
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Hybrid search combining BM25 and embeddings."""
        # Get results from both indices
        bm25_results = self.bm25_index.search(query, top_k * 2)
        embedding_results = self.embedding_index.search(query, top_k * 2)
        
        # Combine scores
        chunk_scores = {}
        
        # Add BM25 scores
        for result in bm25_results:
            chunk_id = id(result.chunk)
            chunk_scores[chunk_id] = {
                'chunk': result.chunk,
                'bm25_score': result.score,
                'embedding_score': 0.0
            }
        
        # Add embedding scores
        for result in embedding_results:
            chunk_id = id(result.chunk)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]['embedding_score'] = result.similarity
            else:
                chunk_scores[chunk_id] = {
                    'chunk': result.chunk,
                    'bm25_score': 0.0,
                    'embedding_score': result.similarity
                }
        
        # Calculate hybrid scores and sort
        hybrid_results = []
        for chunk_data in chunk_scores.values():
            hybrid_score = (
                self.bm25_weight * chunk_data['bm25_score'] +
                self.embedding_weight * chunk_data['embedding_score']
            )
            hybrid_results.append((chunk_data['chunk'], hybrid_score))
        
        # Sort by hybrid score and return top-k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results[:top_k]




