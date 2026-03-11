"""Folder indexing pipeline.

Walk → hash → diff against store → batch-embed new/changed files → upsert.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .models import ModelManager
from .store import ImageStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_standard(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.debug("Failed to open %s: %s", path, e)
        return None


def _load_exr(path: Path) -> Image.Image | None:
    """Load an EXR file, tone-mapping HDR values to 8-bit RGB."""
    try:
        import Imath
        import OpenEXR

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(height, width)
        g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(height, width)
        b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(height, width)

        rgb = np.stack([r, g, b], axis=-1)
        # Reinhard tone mapping: compress HDR values into [0, 1].
        rgb = rgb / (rgb + 1.0)
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")

    except ImportError:
        logger.debug("OpenEXR not installed — skipping %s", path)
    except Exception as e:
        logger.debug("Failed to load EXR %s: %s", path, e)
    return None


def load_image(path: Path) -> Image.Image | None:
    """Load any supported image format as an RGB PIL Image."""
    if path.suffix.lower() == ".exr":
        return _load_exr(path)
    return _load_standard(path)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_hash(path: Path) -> str:
    """Return a blake2b hex digest of the file contents."""
    h = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Folder walking
# ---------------------------------------------------------------------------


def _walk_folder(folder: Path, supported_extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in supported_extensions:
            files.append(f)
    return files


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def index_folders(
    folders: list[str | Path],
    store: ImageStore,
    model: ModelManager,
    supported_extensions: set[str],
    batch_size: int = 32,
) -> dict:
    """Index all given folders into the store.

    Only files whose blake2b hash has changed since the last run are
    re-embedded. Records for files no longer on disk are removed.

    Returns a summary dict with keys: total, skipped, indexed, failed,
    stale_removed.
    """
    folders = [Path(f) for f in folders]

    # 1. Collect all candidate files across all folders.
    all_files: list[Path] = []
    for folder in folders:
        if not folder.exists():
            logger.warning("Folder does not exist, skipping: %s", folder)
            continue
        found = _walk_folder(folder, supported_extensions)
        logger.info("  %s — %d file(s)", folder, len(found))
        all_files.extend(found)

    # Deduplicate by resolved path (handles symlinks / overlapping globs).
    seen: dict[str, Path] = {}
    for f in all_files:
        seen[str(f.resolve())] = f
    all_files = list(seen.values())

    logger.info(
        "Found %d image file(s) across %d folder(s)", len(all_files), len(folders)
    )

    if not all_files:
        return {"total": 0, "skipped": 0, "indexed": 0, "failed": 0, "stale_removed": 0}

    # 2. Load stored hashes and determine what needs re-indexing.
    stored_hashes = store.get_all_hashes()
    valid_paths: set[str] = set()
    to_index: list[tuple[Path, str]] = []

    for f in tqdm(all_files, desc="Hashing", unit="file", leave=False):
        norm_path = f.as_posix()
        valid_paths.add(norm_path)
        try:
            current_hash = compute_hash(f)
        except OSError as e:
            logger.warning("Cannot hash %s: %s", f, e)
            continue
        if stored_hashes.get(norm_path) != current_hash:
            to_index.append((f, current_hash))

    skipped = len(all_files) - len(to_index)
    logger.info(
        "%d file(s) to embed, %d unchanged (skipped)", len(to_index), skipped
    )

    # 3. Remove records for files that have been deleted from disk.
    stale_removed = store.delete_stale(valid_paths)

    if not to_index:
        return {
            "total": len(all_files),
            "skipped": skipped,
            "indexed": 0,
            "failed": 0,
            "stale_removed": stale_removed,
        }

    # 4. Embed and store in batches.
    processed = 0
    failed = 0

    batches = range(0, len(to_index), batch_size)
    for i in tqdm(batches, desc="Embedding", unit="batch"):
        batch = to_index[i : i + batch_size]

        # Load images, skipping any that can't be opened.
        loaded: list[tuple[Path, str, Image.Image]] = []
        for file_path, file_hash in batch:
            img = load_image(file_path)
            if img is None:
                logger.warning("Could not load image, skipping: %s", file_path)
                failed += 1
                continue
            loaded.append((file_path, file_hash, img))

        if not loaded:
            continue

        # Forward pass through the vision encoder.
        try:
            embeddings = model.encode_image([item[2] for item in loaded])
        except Exception as e:
            logger.error("Embedding failed for batch starting at index %d: %s", i, e)
            failed += len(loaded)
            continue

        # Build records and upsert into the store.
        now = time.time()
        records = []
        for (file_path, file_hash, _), embedding in zip(loaded, embeddings):
            records.append(
                {
                    "path": file_path.as_posix(),
                    "file_hash": file_hash,
                    "filename": file_path.name,
                    "ext": file_path.suffix.lower(),
                    "mtime": file_path.stat().st_mtime,
                    "indexed_at": now,
                    "embedding": embedding.tolist(),
                }
            )

        store.upsert(records)
        processed += len(records)

    return {
        "total": len(all_files),
        "skipped": skipped,
        "indexed": processed,
        "failed": failed,
        "stale_removed": stale_removed,
    }
