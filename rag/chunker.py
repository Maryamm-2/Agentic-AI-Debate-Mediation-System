"""Document chunking utilities for RAG system."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    content: str
    source_file: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class DocumentChunker:
    """Handles document chunking for RAG retrieval."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences for better chunk boundaries
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    source_file=source_file,
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={"sentence_count": len(current_chunk.split("."))}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text)
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                source_file=source_file,
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={"sentence_count": len(current_chunk.split("."))}
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_file(self, file_path: Path) -> List[DocumentChunk]:
        """Chunk a single file with robust encoding handling."""
        try:
            # Try multiple encodings in order of likelihood
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    text = file_path.read_text(encoding=encoding)
                    # Validate that we got meaningful text
                    if text and len(text.strip()) > 10:
                        return self.chunk_text(text, str(file_path))
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    print(f"Error reading {file_path} with {encoding}: {e}")
                    continue
            
            # If all encodings failed, try with error handling
            try:
                text = file_path.read_text(encoding='utf-8', errors='replace')
                if text and len(text.strip()) > 10:
                    print(f"Warning: Used error replacement for {file_path}")
                    return self.chunk_text(text, str(file_path))
            except Exception as e:
                print(f"Final attempt failed for {file_path}: {e}")
            
            print(f"Error: Could not read file {file_path} with any encoding")
            return []
            
        except Exception as e:
            print(f"Error chunking file {file_path}: {e}")
            return []
    
    def chunk_directory(self, directory: Path, extensions: List[str] = None) -> List[DocumentChunk]:
        """Chunk all files in a directory."""
        if extensions is None:
            extensions = ['.txt', '.md', '.rst']
        
        all_chunks = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                chunks = self.chunk_file(file_path)
                all_chunks.extend(chunks)
        
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        words = text.split()
        if len(words) <= self.overlap:
            return text
        return " ".join(words[-self.overlap:])



