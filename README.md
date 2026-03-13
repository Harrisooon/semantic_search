# semantic_search

A fully local semantic image search tool for CG artists. Index folders of visual reference and inspiration images, then search them with natural language — no cloud, no API keys, no data leaving your machine.

---

## Overview

`semantic_search` uses [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) to generate vision-language embeddings for every image in your reference library. Those embeddings are stored in [LanceDB](https://lancedb.github.io/lancedb/), a file-based vector database that needs no running daemon. At query time, your text is embedded with the same model and the nearest-neighbor images are returned, ranked by cosine similarity.

Queries like `"moody forest lighting"`, `"hard surface sci-fi panel detail"`, or `"warm backlit portrait"` work as you'd expect — because SigLIP 2 was trained to align image and text representations in the same embedding space.

Everything runs on your local machine. Nothing is sent to an external server.

---

## Features

- **Fully local** — no API keys, no telemetry, no network calls after model download
- **Natural language search** — query your library the way you think about images
- **Find similar** — click any image to find visually similar results using its embedding directly
- **Duplicate filtering** — search and find-similar results deduplicate by file hash, so identical files stored in multiple folders appear only once
- **Fast re-indexing** — blake2b file hashing skips unchanged files on subsequent runs; deleted files are pruned automatically
- **File-based vector store** — LanceDB stores everything in a directory; no database daemon required
- **Web UI** — single-page grid interface with lightbox, copy path, find similar, arrow key navigation, and in-browser reindex trigger
- **Index count** — live image count displayed in the header, updated after every reindex
- **CLI** — scriptable `index`, `search`, and `status` commands
- **GPU-accelerated** — runs on CUDA if available, falls back to CPU automatically
- **CG format support** — PNG, JPG, GIF, TGA, BMP, TIFF, WebP, and EXR (via OpenEXR)
- **GIF preview** — animated GIFs show a static thumbnail in the grid and animate on hover; the lightbox plays them fully animated
- **Live thumbnails** — images resized on-the-fly via Pillow; no stored thumbnail copies

---

## Architecture

```
semantic_search/
├── README.md
├── config.yaml
├── install.bat              # double-click to install with CUDA PyTorch
├── launch.bat               # double-click to start server and open browser
├── semantic_search/
│   ├── __init__.py          # load_config(), path normalisation
│   ├── models.py            # SigLIP 2 loading, encode_image(), encode_text()
│   ├── indexer.py           # folder walking, hash change detection, batch embedding
│   ├── store.py             # LanceDB wrapper: upsert, search, delete stale
│   └── search.py            # text query → vector → ranked results with scores
└── server/
    ├── app.py               # FastAPI backend
    └── templates/
        └── index.html       # single-page search UI
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `models.py` | Loads the SigLIP 2 model and processor once. Exposes `encode_image(pil_images)` and `encode_text(queries)` returning normalised float32 vectors. |
| `indexer.py` | Walks configured folders, computes blake2b hashes per file, diffs against stored hashes, batches new/changed images, calls `encode_image()`, writes to the store. Removes records for deleted files. |
| `store.py` | Thin wrapper around LanceDB. Handles table creation, upsert by file path, similarity search with optional hash-based deduplication, and deletion of stale records. |
| `search.py` | Takes a raw text query, calls `encode_text()`, delegates to `store.search()`, and returns ranked results with file paths and similarity scores. |
| `server/app.py` | FastAPI app. `GET /search?q=` returns JSON results; `GET /count` returns the total indexed image count; `GET /thumb?path=&size=` returns a resized JPEG; `GET /raw?path=` serves the original file (used for animated GIFs); `GET /similar?path=` returns visually similar images by embedding lookup; `POST /reindex` triggers an incremental background reindex. |
| `server/templates/index.html` | Vanilla JS single-page UI. Search bar, responsive image grid, lightbox with copy path and find similar, GIF badge and hover animation. |

---

## Installation

**Requirements:** Python 3.10+, pip, CUDA 12.1-compatible GPU (recommended).

```bash
git clone https://github.com/yourname/semantic_search.git
cd semantic_search
```

**Windows:** double-click `install.bat`. This creates a virtual environment in `.\venv\`, installs PyTorch with CUDA 12.1 support, then installs the remaining dependencies. `launch.bat` activates the venv automatically.

**Other platforms / manual install:**

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[server]"
```

CUDA is the default and strongly recommended — indexing on CPU is significantly slower. If you want the CPU-only build instead, replace the torch install line with:

```bash
pip install torch torchvision
```

The SigLIP 2 model weights (~400 MB) are downloaded from Hugging Face on first run and cached in `~/.cache/huggingface/`.

### OpenEXR support

EXR files require the `OpenEXR` Python bindings, which need the OpenEXR C++ library installed on your system first.

```bash
# Debian/Ubuntu
sudo apt install libopenexr-dev
pip install openexr

# macOS (Homebrew)
brew install openexr
pip install openexr

# Windows — install via vcpkg or use a pre-built wheel.
# If unavailable, EXR files are silently skipped.
```

---

## Configuration

Edit `config.yaml` before first use:

```yaml
# Folders to index. Supports ~ expansion. Multiple folders share one index.
watched_folders:
  - D:/references

# LanceDB storage directory.
db_path: ~/.semantic_search/db

# Model variant.
model_variant: google/siglip2-base-patch16-224

# Device: "auto" selects CUDA if available, otherwise CPU.
# Force with "cuda", "cuda:1", or "cpu".
device: auto

# Number of images processed per model forward pass.
# Increase for better GPU utilization; decrease if you run out of VRAM.
batch_size: 32

# File extensions to index. Case-insensitive.
supported_extensions:
  - .gif
  - .png
  - .jpg
  - .jpeg
  - .tga
  - .bmp
  - .tiff
  - .webp
  - .exr
```

---

## Usage

### Web UI

Double-click `launch.bat`, or start the server manually:

```bash
python server/app.py
```

Then open `http://localhost:8000` in your browser. The server loads the model once and keeps it in memory for the duration of the session.

**Search:** type a natural language query in the search bar. Results appear as a responsive image grid sorted by similarity score.

**Lightbox:** click any image to open a full-size view with copy path and find similar buttons. Use the left/right arrow keys to navigate between results.

**Find similar:** hover a card and click "find similar", or use the button in the lightbox, to search by visual embedding rather than text.

**GIFs:** display as static thumbnails in the grid, animate on hover, and play fully in the lightbox.

**Reindex:** click the reindex button in the header to scan your watched folders for new, changed, or deleted files without restarting the server. The current search results refresh automatically when the reindex completes.

### CLI

```bash
semantic-search index        # index/reindex all watched_folders from config.yaml
semantic-search status       # show count, per-folder breakdown, and last-indexed time
semantic-search search "moody forest lighting"
semantic-search search "hard surface sci-fi panel detail" --top 20
```

Re-running `index` is safe and fast — only files whose content has changed are re-embedded. Deleted files are pruned from the index automatically.

---

## How It Works

### Indexing pipeline

1. **Folder walk** — `indexer.py` recursively walks each folder in `watched_folders`, filtered by `supported_extensions`.
2. **Hash check** — For each file, a blake2b hash of the file contents is computed. If the hash matches the stored hash for that path, the file is skipped.
3. **Batch loading** — New and changed files are grouped into batches of `batch_size` and loaded as PIL Images. EXR files are tone-mapped to 8-bit RGB via Reinhard mapping. GIFs are embedded using their first frame.
4. **Embedding** — Each batch is passed through SigLIP 2's image encoder. The model and processor are loaded once at startup and reused for all batches.
5. **Storage** — Embeddings, file paths, hashes, and metadata are upserted into LanceDB. Records for deleted files are removed.

### Search pipeline

1. **Text encoding** — The query string is tokenised and passed through SigLIP 2's text encoder, producing a normalised embedding vector.
2. **ANN search** — `store.py` runs an approximate nearest-neighbour search over the image embeddings using cosine distance.
3. **Deduplication** — Results are filtered by `file_hash`: if multiple paths share the same hash, only the highest-scoring one is returned.
4. **Result ranking** — Results are returned sorted by similarity score (1.0 = identical). The top-k results are returned as `(path, score)` pairs.
5. **Serving** — The web UI fetches results from `/search`, then requests thumbnails via `/thumb` (JPEG, resized on-the-fly). GIFs are served raw via `/raw` to preserve animation.

### Find similar pipeline

1. The stored embedding for the selected image is retrieved directly from LanceDB.
2. An ANN search is run using that embedding as the query vector (same pipeline as text search).
3. The query image itself is excluded from results by path comparison.
4. Duplicate images are filtered by hash, same as text search.

---

## Performance Tips

- **GPU is strongly recommended.** Indexing 10,000 images on CPU takes on the order of hours; on a modern GPU it takes minutes.
- **Increase `batch_size`** if your GPU has headroom. For a 12 GB card, try 64–128. Decrease if you hit CUDA out-of-memory errors.
- **Re-indexing is incremental.** After the initial index build, subsequent runs only process new or changed files.
- **EXR files are slower to load** than other formats due to format conversion overhead. If you have large EXR sequences, consider indexing exported JPEG/PNG derivatives instead.
- **Model loading takes ~2–5 seconds.** The CLI reloads the model on every invocation. For repeated searches, use the web UI — the server loads the model once and keeps it in memory.

---

## License

MIT
