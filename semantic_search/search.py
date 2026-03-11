"""Text query → ranked image results."""

from __future__ import annotations

import logging
from pathlib import Path

from .models import ModelManager
from .store import ImageStore

logger = logging.getLogger(__name__)


def search(
    query: str,
    store: ImageStore,
    model: ModelManager,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Search the index with a natural language query.

    Args:
        query:  Natural language search string, e.g. "moody forest lighting".
        store:  Initialised ImageStore pointing at the LanceDB directory.
        model:  Loaded ModelManager (shared with indexer to avoid reloading).
        top_k:  Maximum number of results to return.

    Returns:
        List of (path, score) tuples, sorted by descending cosine similarity.
        Paths use the OS path separator. Score is in [−1, 1]; higher = better.
    """
    if not query.strip():
        return []

    vector = model.encode_text([query])[0]
    results = store.search(vector, top_k=top_k)

    # Convert stored posix paths back to OS-native paths.
    results = [(str(Path(p)), score) for p, score in results]

    logger.debug("'%s' → %d result(s), top score %.4f", query, len(results), results[0][1] if results else 0)
    return results
