"""
knowledge_base.py — Persistent multi-document knowledge base for the RAG system.

Manages PDF documents, chunk metadata, and FAISS index persistence.
Reuses all existing pipeline_utils functions for extraction, cleaning, chunking, and retrieval.

Usage:
    from knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb.add_document("/path/to/file.pdf", "filename.pdf", embedding_model)
    results = kb.search("query", embedding_model, k=10)
"""

import os
import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

from pipeline_utils import (
    clean_page_text,
    extract_chapter_titles,
    find_chapter_start_pages,
    assign_chapter_labels,
    chunk_text,
    build_context_package,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("knowledge_base")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # BAAI/bge-base-en-v1.5

METADATA_COLUMNS = [
    "document_id", "document_name", "page_number",
    "chapter", "chunk_id", "chunk_text", "search_text",
]

DOC_COLUMNS = [
    "document_id", "document_name", "num_pages",
    "num_chunks", "upload_time", "file_size",
]


# ---------------------------------------------------------------------------
# KnowledgeBase class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Persistent multi-document knowledge base backed by FAISS and parquet."""

    def __init__(self, base_dir=None):
        """Initialize paths and load existing data from disk."""
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent / "knowledge_base"
        self.base_dir = Path(base_dir)

        self.documents_dir = self.base_dir / "documents"
        self.indexes_dir = self.base_dir / "indexes"
        self.metadata_dir = self.base_dir / "metadata"

        self.faiss_path = self.indexes_dir / "faiss.index"
        self.chunks_path = self.metadata_dir / "chunks.parquet"
        self.docs_path = self.metadata_dir / "documents.csv"

        # Create directories
        for d in [self.documents_dir, self.indexes_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self.documents_df = pd.DataFrame(columns=DOC_COLUMNS)
        self.chunks_df = pd.DataFrame(columns=METADATA_COLUMNS)
        self.faiss_index = None

        # Load existing data
        self.load()

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def load(self):
        """Load all knowledge base data from disk."""
        self._load_documents()
        self._load_chunks()
        self._load_faiss()
        self._validate_integrity()

    def _load_documents(self):
        """Load document registry from CSV."""
        if self.docs_path.exists():
            try:
                self.documents_df = pd.read_csv(self.docs_path)
                # Ensure all required columns exist
                for col in DOC_COLUMNS:
                    if col not in self.documents_df.columns:
                        self.documents_df[col] = None
                logger.info(f"Loaded {len(self.documents_df)} document records")
            except Exception as e:
                logger.warning(f"Error loading documents.csv: {e}. Starting fresh.")
                self.documents_df = pd.DataFrame(columns=DOC_COLUMNS)
        else:
            self.documents_df = pd.DataFrame(columns=DOC_COLUMNS)

    def _load_chunks(self):
        """Load chunk metadata from parquet."""
        if self.chunks_path.exists():
            try:
                self.chunks_df = pd.read_parquet(self.chunks_path)
                for col in METADATA_COLUMNS:
                    if col not in self.chunks_df.columns:
                        self.chunks_df[col] = None
                logger.info(f"Loaded {len(self.chunks_df)} chunk records")
            except Exception as e:
                logger.warning(f"Error loading chunks.parquet: {e}. Starting fresh.")
                self.chunks_df = pd.DataFrame(columns=METADATA_COLUMNS)
        else:
            self.chunks_df = pd.DataFrame(columns=METADATA_COLUMNS)

    def _load_faiss(self):
        """Load FAISS index from disk."""
        if self.faiss_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Error loading FAISS index: {e}. Will rebuild if needed.")
                self.faiss_index = None
        else:
            self.faiss_index = None

    def _validate_integrity(self):
        """Validate data integrity: remove orphaned docs and chunks."""
        if self.documents_df.empty and self.chunks_df.empty:
            return

        doc_ids = set(self.documents_df["document_id"].values) if not self.documents_df.empty else set()
        chunk_doc_ids = set(self.chunks_df["document_id"].values) if not self.chunks_df.empty else set()

        # Remove chunks whose document no longer exists
        if not self.chunks_df.empty and chunk_doc_ids - doc_ids:
            orphaned = chunk_doc_ids - doc_ids
            logger.warning(f"Removing {len(orphaned)} orphaned chunk sets")
            self.chunks_df = self.chunks_df[~self.chunks_df["document_id"].isin(orphaned)]

        # Remove documents whose chunks no longer exist
        if not self.documents_df.empty and doc_ids - chunk_doc_ids:
            orphaned = doc_ids - chunk_doc_ids
            logger.warning(f"Removing {len(orphaned)} orphaned document records")
            self.documents_df = self.documents_df[~self.documents_df["document_id"].isin(orphaned)]

        # Rebuild FAISS if vector count doesn't match chunk count
        if self.faiss_index is not None:
            if self.chunks_df.empty and self.faiss_index.ntotal > 0:
                logger.warning("FAISS has vectors but no chunks exist. Clearing FAISS.")
                self.faiss_index = None
            elif not self.chunks_df.empty and self.faiss_index.ntotal != len(self.chunks_df):
                logger.warning(
                    f"FAISS vectors ({self.faiss_index.ntotal}) != chunks ({len(self.chunks_df)}). "
                    "Clearing stale FAISS index."
                )
                self.faiss_index = None

    def save(self):
        """Persist all data to disk."""
        # Save documents CSV
        if not self.documents_df.empty:
            self.documents_df.to_csv(self.docs_path, index=False)
        else:
            # Write empty CSV with headers
            pd.DataFrame(columns=DOC_COLUMNS).to_csv(self.docs_path, index=False)

        # Save chunks parquet
        if not self.chunks_df.empty:
            self.chunks_df.to_parquet(self.chunks_path, index=False)
        else:
            # Write empty parquet with headers
            pd.DataFrame(columns=METADATA_COLUMNS).to_parquet(self.chunks_path, index=False)

        # Save FAISS index (or remove stale file)
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.faiss_path))
        elif self.faiss_path.exists():
            # Remove stale FAISS file when index is None (e.g., after deletion)
            self.faiss_path.unlink()

        logger.info("Knowledge base saved to disk")

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    @staticmethod
    def get_file_hash(file_path):
        """Compute SHA256 hash of a file for duplicate detection."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def is_empty(self):
        """Check if the knowledge base has no documents."""
        return self.documents_df.empty

    def document_exists(self, file_hash):
        """Check if a document with this hash already exists."""
        if self.documents_df.empty:
            return False
        return file_hash in self.documents_df["document_id"].values

    # -------------------------------------------------------------------
    # Add document
    # -------------------------------------------------------------------

    def add_document(self, file_path, file_name, embedding_model):
        """
        Process and add a new PDF document to the knowledge base.

        Steps:
        1. Check for duplicates via SHA256
        2. Extract text using PyMuPDF (via existing pipeline_utils)
        3. Clean text using existing clean_page_text
        4. Assign chapter labels using existing chapter functions
        5. Chunk using existing chunk_text
        6. Generate embeddings using the provided model
        7. Append embeddings to FAISS index
        8. Append chunk metadata to chunks DataFrame
        9. Save everything to disk

        Returns: dict with keys: success, document_id, num_pages, num_chunks, message
        """
        import fitz

        file_path = Path(file_path)

        # 1. Duplicate check
        file_hash = self.get_file_hash(file_path)
        if self.document_exists(file_hash):
            return {
                "success": False,
                "document_id": file_hash,
                "message": f"Document '{file_name}' already exists in the knowledge base (duplicate detected).",
            }

        logger.info(f"Processing document: {file_name} (hash: {file_hash[:12]}...)")

        # 2. Extract text
        pdf_document = fitz.open(str(file_path))
        raw_pages = []
        for page in pdf_document:
            raw_pages.append(page.get_text())

        # 3. Chapter detection and labels (existing functions)
        chapter_titles = extract_chapter_titles(pdf_document, toc_page_range=range(0, min(8, pdf_document.page_count)))
        chapter_start_pages = find_chapter_start_pages(pdf_document)
        chapter_labels = assign_chapter_labels(
            pdf_document.page_count, chapter_start_pages, chapter_titles
        )

        # Build page DataFrame
        pages_df = pd.DataFrame({
            "page_number": range(pdf_document.page_count),
            "raw_text": raw_pages,
            "chapter": chapter_labels,
        })

        # 4. Clean text (existing function)
        pages_df["clean_text"] = pages_df["raw_text"].apply(clean_page_text)
        pages_df["clean_word_count"] = pages_df["clean_text"].apply(lambda t: len(t.split()))

        # Filter pages: remove front matter and short pages.
        # Only apply the "Front Matter" filter if chapter detection actually found
        # at least one chapter — otherwise every page is labeled "Front Matter" and
        # the entire document would be discarded (e.g. PDFs with different formatting).
        if chapter_start_pages:
            pages_df = pages_df[pages_df["chapter"] != "Front Matter"].reset_index(drop=True)
        pages_df = pages_df[pages_df["clean_word_count"] >= 40].reset_index(drop=True)

        if len(pages_df) == 0:
            pdf_document.close()
            return {
                "success": False,
                "document_id": file_hash,
                "message": f"No usable content found in '{file_name}' after filtering.",
            }

        # 5. Chunk (existing function)
        doc_short_id = file_hash[:12]
        chunk_rows = []
        for _, row in pages_df.iterrows():
            page_chunks = chunk_text(row["clean_text"], chunk_size=100, overlap=25)
            for i, chunk in enumerate(page_chunks):
                chunk_rows.append({
                    "document_id": file_hash,
                    "document_name": file_name,
                    "page_number": row["page_number"],
                    "chapter": row["chapter"],
                    "chunk_id": f"{doc_short_id}_page{row['page_number']}_chunk{i}",
                    "chunk_text": chunk,
                    "search_text": f"{row['chapter']} : {chunk}",
                })

        if not chunk_rows:
            pdf_document.close()
            return {
                "success": False,
                "document_id": file_hash,
                "message": f"No chunks generated from '{file_name}'.",
            }

        new_chunks_df = pd.DataFrame(chunk_rows)

        # 6. Generate embeddings
        logger.info(f"Generating embeddings for {len(new_chunks_df)} chunks...")
        embeddings = embedding_model.encode(
            new_chunks_df["search_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # 7. Append to FAISS
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        # 8. Append chunk metadata
        if self.chunks_df.empty:
            self.chunks_df = new_chunks_df
        else:
            self.chunks_df = pd.concat([self.chunks_df, new_chunks_df], ignore_index=True)

        # 9. Save PDF to documents directory
        doc_dir = self.documents_dir / doc_short_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        saved_pdf_path = doc_dir / file_name
        shutil.copy2(str(file_path), str(saved_pdf_path))

        # 10. Update document registry
        new_doc = pd.DataFrame([{
            "document_id": file_hash,
            "document_name": file_name,
            "num_pages": len(pages_df),
            "num_chunks": len(new_chunks_df),
            "upload_time": datetime.now().isoformat(),
            "file_size": file_path.stat().st_size,
        }])
        if self.documents_df.empty:
            self.documents_df = new_doc
        else:
            self.documents_df = pd.concat([self.documents_df, new_doc], ignore_index=True)

        # 11. Persist to disk
        self.save()

        pdf_document.close()

        logger.info(
            f"Document added: {file_name} — {len(pages_df)} pages, "
            f"{len(new_chunks_df)} chunks, FAISS total: {self.faiss_index.ntotal}"
        )

        return {
            "success": True,
            "document_id": file_hash,
            "num_pages": len(pages_df),
            "num_chunks": len(new_chunks_df),
            "message": f"'{file_name}' indexed successfully ({len(pages_df)} pages, {len(new_chunks_df)} chunks).",
        }

    # -------------------------------------------------------------------
    # Delete document
    # -------------------------------------------------------------------

    def delete_document(self, document_id):
        """
        Remove a document and all its chunks from the knowledge base.

        Steps:
        1. Remove chunks for this document
        2. Remove document from registry
        3. Delete stored PDF
        4. Rebuild FAISS from remaining chunks (IndexFlatIP doesn't support removal)
        5. Save to disk

        Returns: dict with success, message
        """
        if self.documents_df.empty or document_id not in self.documents_df["document_id"].values:
            return {"success": False, "message": "Document not found."}

        doc_row = self.documents_df[self.documents_df["document_id"] == document_id].iloc[0]
        doc_name = doc_row["document_name"]
        doc_short_id = document_id[:12]

        logger.info(f"Deleting document: {doc_name} ({doc_short_id}...)")

        # 1. Remove chunks
        self.chunks_df = self.chunks_df[self.chunks_df["document_id"] != document_id].reset_index(drop=True)

        # 2. Remove from document registry
        self.documents_df = self.documents_df[self.documents_df["document_id"] != document_id].reset_index(drop=True)

        # 3. Delete stored PDF
        doc_dir = self.documents_dir / doc_short_id
        if doc_dir.exists():
            shutil.rmtree(str(doc_dir))

        # 4. Rebuild FAISS from remaining chunks
        if self.chunks_df.empty:
            self.faiss_index = None
            logger.info("All documents deleted. FAISS index cleared.")
        else:
            logger.info(f"Rebuilding FAISS index from {len(self.chunks_df)} remaining chunks...")
            # We need the embedding model to rebuild — store a flag and let the caller handle it
            # Actually, we can't rebuild without the model. Set to None and let the app rebuild.
            self.faiss_index = None
            self._needs_faiss_rebuild = True
            logger.info("FAISS index cleared. Will be rebuilt on next query.")

        # 5. Save
        self.save()

        logger.info(f"Document deleted: {doc_name}")

        return {
            "success": True,
            "message": f"'{doc_name}' deleted successfully.",
        }

    def rebuild_faiss(self, embedding_model):
        """
        Rebuild the FAISS index from all remaining chunks.
        Called after deletion or when the index is out of sync.
        """
        if self.chunks_df.empty:
            self.faiss_index = None
            self.save()
            return

        logger.info(f"Rebuilding FAISS index from {len(self.chunks_df)} chunks...")
        embeddings = embedding_model.encode(
            self.chunks_df["search_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        self.save()
        logger.info(f"FAISS index rebuilt: {self.faiss_index.ntotal} vectors")

    @property
    def needs_faiss_rebuild(self):
        """Check if the FAISS index needs rebuilding (e.g., after deletion)."""
        if self.faiss_index is None and not self.chunks_df.empty:
            return True
        if self.faiss_index is not None and self.faiss_index.ntotal != len(self.chunks_df):
            return True
        return False

    # -------------------------------------------------------------------
    # Search / Retrieval
    # -------------------------------------------------------------------

    def search(self, query, embedding_model, k=10):
        """
        Search across ALL documents in the knowledge base.

        Returns a DataFrame with columns:
            document_id, document_name, page_number, chapter,
            chunk_id, chunk_text, score
        """
        if self.faiss_index is None or self.chunks_df.empty:
            return pd.DataFrame(columns=METADATA_COLUMNS + ["score"])

        try:
            query_embedding = embedding_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")

            actual_k = min(k, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_embedding, actual_k)

            results = self.chunks_df.iloc[indices[0]].copy()
            results["score"] = scores[0]

            return results.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Search error: {e}")
            return pd.DataFrame(columns=METADATA_COLUMNS + ["score"])

    def build_context_package(self, query, embedding_model,
                              retrieval_k=10, max_context_chunks=4,
                              max_chunks_per_page=2, word_budget=220):
        """
        Build a context package for the LLM — same logic as pipeline_utils.build_context_package
        but works across all documents.

        Returns dict with: query, candidates, selected, context_text, used_words, num_sources
        """
        candidates = self.search(query, embedding_model, k=retrieval_k)

        if candidates.empty:
            return {
                "query": query,
                "candidates": candidates,
                "selected": pd.DataFrame(),
                "context_text": "",
                "used_words": 0,
                "num_sources": 0,
            }

        import re
        selected_rows = []
        seen_texts = set()
        per_page_counts = {}
        used_words = 0

        for _, row in candidates.iterrows():
            normalized = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
            if normalized in seen_texts:
                continue

            # Use (document_id, page_number) as key to allow same page numbers across docs
            page_key = (row["document_id"], row["page_number"])
            page_count = per_page_counts.get(page_key, 0)
            if page_count >= max_chunks_per_page:
                continue

            chunk_words = len(row["chunk_text"].split())
            if selected_rows and used_words + chunk_words > word_budget:
                continue

            selected_rows.append(row.to_dict())
            seen_texts.add(normalized)
            per_page_counts[page_key] = page_count + 1
            used_words += chunk_words

            if len(selected_rows) >= max_context_chunks:
                break

        # Build context text with document names
        blocks = []
        for position, row in enumerate(selected_rows, start=1):
            doc_name = row.get("document_name", "Unknown")
            blocks.append(
                f"[Source {position}] (Document: {doc_name}, Page {row['page_number']}, {row['chapter']})\n{row['chunk_text']}"
            )

        return {
            "query": query,
            "candidates": candidates,
            "selected": pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(),
            "context_text": "\n\n".join(blocks),
            "used_words": used_words,
            "num_sources": len(selected_rows),
        }

    # -------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------

    def get_stats(self):
        """Return knowledge base statistics."""
        total_docs = len(self.documents_df) if not self.documents_df.empty else 0
        total_pages = int(self.documents_df["num_pages"].sum()) if not self.documents_df.empty else 0
        total_chunks = len(self.chunks_df) if not self.chunks_df.empty else 0
        total_vectors = self.faiss_index.ntotal if self.faiss_index is not None else 0

        return {
            "total_documents": total_docs,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors,
        }

    def get_all_documents(self):
        """Return all documents as a DataFrame, sorted by upload time."""
        if self.documents_df.empty:
            return pd.DataFrame(columns=DOC_COLUMNS)
        return self.documents_df.sort_values("upload_time", ascending=False).reset_index(drop=True)

    def get_document_info(self, document_id):
        """Get metadata for a specific document."""
        if self.documents_df.empty:
            return None
        matches = self.documents_df[self.documents_df["document_id"] == document_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    # -------------------------------------------------------------------
    # Migration: Import existing PDF into knowledge base
    # -------------------------------------------------------------------

    def migrate_existing_pdf(self, pdf_path, embedding_model):
        """
        One-time migration: import the existing textbook PDF into the knowledge base.
        Only runs if the knowledge base is empty.

        Returns: dict with success, message
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"success": False, "message": f"PDF not found: {pdf_path}"}

        file_hash = self.get_file_hash(pdf_path)
        if self.document_exists(file_hash):
            return {"success": False, "message": "Already migrated."}

        logger.info(f"Migrating existing PDF: {pdf_path.name}")
        return self.add_document(pdf_path, pdf_path.name, embedding_model)
