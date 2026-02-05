"""BM25 keyword-based retrieval system."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
import numpy as np

from .chunker import DocumentChunk


@dataclass
class RetrievalResult:
    """Result from BM25 retrieval."""
    chunk: DocumentChunk
    score: float
    rank: int


class BM25Index:
    """BM25-based keyword retrieval index."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_chunks: List[List[str]] = []
    
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index from document chunks."""
        self.chunks = chunks
        self.tokenized_chunks = [self._tokenize(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks, k1=self.k1, b=self.b)
        print(f"Built BM25 index with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search the index for relevant chunks."""
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only return chunks with positive scores
                result = RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank
                )
                results.append(result)
        
        return results
    
    def save_index(self, path: Path) -> None:
        """Save the BM25 index to disk."""
        index_data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_chunks': self.tokenized_chunks,
            'k1': self.k1,
            'b': self.b
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Saved BM25 index to {path}")
    
    def load_index(self, path: Path) -> bool:
        """Load BM25 index from disk."""
        try:
            with open(path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25 = index_data['bm25']
            self.chunks = index_data['chunks']
            self.tokenized_chunks = index_data['tokenized_chunks']
            self.k1 = index_data['k1']
            self.b = index_data['b']
            
            print(f"Loaded BM25 index from {path} with {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            return False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split on whitespace and punctuation."""
        import re
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.bm25:
            return {}
        
        return {
            'num_chunks': len(self.chunks),
            'avg_chunk_length': np.mean([len(chunk.content) for chunk in self.chunks]),
            'vocab_size': len(self.bm25.idf),
            'avg_tokens_per_chunk': np.mean([len(tokens) for tokens in self.tokenized_chunks])
        }




