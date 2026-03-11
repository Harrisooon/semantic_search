"""LanceDB wrapper.

Handles table creation, upsert-by-path, vector search, stale-record cleanup,
and stats. All file paths are stored with forward slashes for cross-platform
consistency; callers should use Path(result_path) to get an OS-native path.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

_TABLE_NAME = "images"


class ImageStore:
    """Wraps a LanceDB table that holds image embeddings and metadata."""

    def __init__(self, db_path: str | Path, embedding_dim: int = 768) -> None:
        import lancedb

        self.db_path = Path(db_path).expanduser()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim

        self._db = lancedb.connect(str(self.db_path))
        self._table = None  # created lazily on first access

    # ------------------------------------------------------------------
    # Schema and table access
    # ------------------------------------------------------------------

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("path", pa.utf8()),
                pa.field("file_hash", pa.utf8()),
                pa.field("filename", pa.utf8()),
                pa.field("ext", pa.utf8()),
                pa.field("mtime", pa.float64()),
                pa.field("indexed_at", pa.float64()),
                pa.field(
                    "embedding",
                    pa.list_(pa.float32(), self.embedding_dim),
                ),
            ]
        )

    @property
    def table(self):
        if self._table is None:
            self._table = self._db.create_table(
                _TABLE_NAME,
                schema=self._schema(),
                exist_ok=True,
            )
        return self._table

    # ------------------------------------------------------------------
    # Read helpers — column-selective scans via the underlying lance dataset
    # to avoid loading large embedding vectors unnecessarily.
    # ------------------------------------------------------------------

    def count(self) -> int:
        try:
            return self.table.count_rows()
        except Exception:
            return 0

    def _read_columns(self, columns: list[str]) -> dict[str, list]:
        """Read specific columns, returning {column_name: [values]} dict."""
        # to_arrow() with column selection — efficient and works with current lancedb
        try:
            tbl = self.table.to_arrow().select(columns)
            return {col: tbl[col].to_pylist() for col in columns}
        except Exception as e:
            logger.warning("to_arrow() failed (%s), falling back to to_pandas()", e)

        # Fallback: pandas loads all columns including embeddings, but always works
        df = self.table.to_pandas()
        return {col: df[col].tolist() for col in columns}

    def get_all_hashes(self) -> dict[str, str]:
        """Return {path: file_hash} for every indexed file."""
        if self.count() == 0:
            return {}
        try:
            data = self._read_columns(["path", "file_hash"])
            result = dict(zip(data["path"], data["file_hash"]))
            logger.info("Loaded %d stored hashes", len(result))
            return result
        except Exception as e:
            logger.warning("Could not read hashes from store: %s", e)
            return {}

    def get_all_paths(self) -> set[str]:
        """Return the set of all indexed paths."""
        if self.count() == 0:
            return set()
        try:
            data = self._read_columns(["path"])
            return set(data["path"])
        except Exception as e:
            logger.warning("Could not read paths from store: %s", e)
            return set()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, records: list[dict[str, Any]]) -> None:
        """Insert or update records, keyed on 'path'."""
        if not records:
            return
        (
            self.table.merge_insert("path")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )

    def delete_stale(self, valid_paths: set[str]) -> int:
        """Delete records for files that no longer exist on disk.

        Args:
            valid_paths: set of normalised (posix) paths that are still present.

        Returns:
            Number of records removed.
        """
        stored = self.get_all_paths()
        stale = stored - valid_paths
        if not stale:
            return 0

        stale_list = list(stale)
        for i in range(0, len(stale_list), 100):
            batch = stale_list[i : i + 100]
            # Escape single quotes in paths to avoid breaking the SQL expression.
            escaped = [p.replace("'", "''") for p in batch]
            expr = "path IN (" + ", ".join(f"'{p}'" for p in escaped) + ")"
            self.table.delete(expr)

        logger.info("Removed %d stale records", len(stale))
        return len(stale)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Approximate nearest-neighbour search by cosine similarity.

        Args:
            vector: L2-normalised float32 array of shape (D,).
            top_k:  number of results to return.

        Returns:
            List of (path, score) sorted by descending score.
            Score is cosine similarity in [−1, 1]; 1.0 = identical.
        """
        results = (
            self.table.search(vector.tolist(), vector_column_name="embedding")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        # LanceDB cosine metric returns distance = 1 − similarity.
        return [(r["path"], round(1.0 - r["_distance"], 4)) for r in results]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return per-folder counts, total, and last-indexed timestamp."""
        if self.count() == 0:
            return {"total": 0, "by_folder": {}, "last_indexed": None}
        try:
            data  = self._read_columns(["path", "indexed_at"])
            paths = data["path"]
            times = data["indexed_at"]
        except Exception as e:
            logger.warning("Could not read stats: %s", e)
            return {"total": 0, "by_folder": {}, "last_indexed": None}

        by_folder = Counter(str(Path(p).parent) for p in paths)
        return {
            "total": len(paths),
            "by_folder": dict(by_folder),
            "last_indexed": max(times) if times else None,
        }
