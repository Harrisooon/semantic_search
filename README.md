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
- **Fast re-indexing** — blake2b file hashing skips unchanged files on subsequent runs
- **File-based vector store** — LanceDB stores everything in a directory; no database daemon required
- **Web UI** — single-page grid interface served by a local FastAPI server
- **CLI** — scriptable `index`, `search`, and `status` commands
- **GPU-accelerated** — runs on CUDA if available, falls back to CPU automatically
- **CG format support** — PNG, JPG, TGA, BMP, TIFF, WebP, and EXR (via OpenEXR fallback)
- **Live thumbnails** — images resized on-the-fly via Pillow; no stored thumbnail copies

---

## Architecture

```
semantic_search/
├── README.md
├── requirements.txt
├── config.yaml
├── refsearch/
│   ├── __init__.py
│   ├── models.py        # SigLIP 2 loading, encode_image(), encode_text()
│   ├── indexer.py       # folder walking, file hash change detection, batch embedding
│   ├── store.py         # LanceDB wrapper: insert, search, delete stale
│   ├── search.py        # text query → vector → ranked results with scores
│   └── watcher.py       # optional filesystem watcher for auto-reindex
├── server/
│   ├── app.py           # FastAPI backend: /search, /thumb endpoints
│   └── templates/
│       └── index.html   # single-page search UI with grid results
└── cli.py               # CLI: refsearch index, refsearch search, refsearch status
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `models.py` | Loads the SigLIP 2 model and processor once. Exposes `encode_image(pil_image)` and `encode_text(query_string)` returning normalized float32 vectors. |
| `indexer.py` | Walks configured folders, computes blake2b hashes per file, diffs against stored hashes, batches new/changed images, calls `encode_image()`, writes to the store. |
| `store.py` | Thin wrapper around LanceDB. Handles table creation, upsert by file path, similarity search, and deletion of records for files that no longer exist on disk. |
| `search.py` | Takes a raw text query, calls `encode_text()`, delegates to `store.search()`, and returns ranked results with file paths and similarity scores. |
| `watcher.py` | Uses `watchdog` to monitor configured folders and trigger incremental re-indexing when files are created, modified, or deleted. Optional — not required for basic use. |
| `server/app.py` | FastAPI app with two endpoints: `GET /search?q=...` returns JSON results; `GET /thumb?path=...&size=...` returns a resized JPEG of any indexed image. |
| `server/templates/index.html` | Vanilla JS single-page UI. Sends queries to `/search`, renders results as a responsive image grid, links through to `/thumb` for display. |
| `cli.py` | Entry point for `refsearch index <path>`, `refsearch search "query"`, and `refsearch status`. Reads config, delegates to the appropriate module. |

---

## Installation

**Requirements:** Python 3.10+, pip.

```bash
git clone https://github.com/yourname/semantic_search.git
cd semantic_search
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU support, install the CUDA-enabled PyTorch build first (before the other requirements), matching your CUDA version:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The SigLIP 2 model weights (~400 MB) are downloaded from Hugging Face on first run and cached in the standard `~/.cache/huggingface/` directory.

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
# Folders to index. Absolute paths. Multiple folders are merged into one index.
watched_folders:
  - /mnt/art/references
  - /mnt/art/mood_boards

# LanceDB storage directory.
db_path: ~/.semantic_search/db

# Model variant. Currently only siglip2-base-patch16-224 is supported.
model_variant: google/siglip2-base-patch16-224

# Device: "auto" selects CUDA if available, otherwise CPU.
# Force with "cuda", "cuda:1", or "cpu".
device: auto

# Number of images processed per model forward pass.
# Increase for better GPU utilization; decrease if you run out of VRAM.
batch_size: 32

# File extensions to index. Case-insensitive.
supported_extensions:
  - .png
  - .jpg
  - .jpeg
  - .tga
  - .bmp
  - .tiff
  - .webp
  - .exr
```

All paths in `config.yaml` support `~` expansion.

---

## Usage

### CLI

**Index a folder** (or all configured `watched_folders`):

```bash
python cli.py index /path/to/your/references
# or index all watched_folders from config.yaml:
python cli.py index
```

Re-running `index` is safe and fast — only files whose content has changed (detected via blake2b hash) are re-embedded.

**Search from the command line:**

```bash
python cli.py search "moody forest lighting"
python cli.py search "hard surface sci-fi panel detail" --top 20
python cli.py search "warm backlit portrait" --top 5
```

Output is a ranked list of file paths with cosine similarity scores:

```
0.847  /mnt/art/references/forests/misty_morning_03.jpg
0.831  /mnt/art/references/env/dark_forest_concept.png
0.809  /mnt/art/mood_boards/lighting_refs/overcast_treeline.tiff
...
```

**Check index status:**

```bash
python cli.py status
```

Shows total indexed files, per-folder counts, last index timestamp, and any files present on disk but not yet indexed.

### Web UI

Start the local server:

```bash
python server/app.py
```

Then open `http://localhost:8000` in your browser.

Type a query in the search bar. Results appear as a responsive image grid, sorted by similarity. Click any image to open the original file path.

The server does not need to be restarted when you re-index — the LanceDB store is read on each query.

### Filesystem watcher (optional)

To automatically re-index when files are added or changed:

```bash
python -m refsearch.watcher
```

This runs in the foreground and processes change events incrementally. Useful if you're actively dropping new references into watched folders.

---

## How It Works

### Indexing pipeline

1. **Folder walk** — `indexer.py` recursively walks each folder in `watched_folders`, filtered by `supported_extensions`.
2. **Hash check** — For each file, a blake2b hash of the file contents is computed. If the hash matches the stored hash for that path, the file is skipped.
3. **Batch loading** — New and changed files are grouped into batches of `batch_size` and loaded as PIL Images. EXR files are handled separately via the OpenEXR library and converted to a float32 PIL Image before being passed to the same pipeline.
4. **Embedding** — Each batch is passed through SigLIP 2's image encoder via `encode_image()`. The model and processor are loaded once at startup and reused for all batches.
5. **Storage** — Embeddings, file paths, hashes, and metadata (filename, extension, modification time) are upserted into LanceDB. Records for files that have been deleted from disk are removed.

### Search pipeline

1. **Text encoding** — The query string is tokenized and passed through SigLIP 2's text encoder via `encode_text()`, producing a normalized embedding vector.
2. **ANN search** — `store.py` runs an approximate nearest-neighbor search over the image embeddings in LanceDB using cosine distance.
3. **Result ranking** — Results are returned sorted by similarity score (1.0 = identical, lower = less similar). The top-k results are returned as a list of `(path, score)` pairs.
4. **Serving** — The web UI fetches results from `/search`, then requests each image via `/thumb?path=<path>&size=<px>`, where the FastAPI server opens the file, resizes it with Pillow to the requested size, and returns it as a JPEG — no thumbnail copies are stored on disk.

---

## Performance Tips

- **GPU is strongly recommended.** Indexing 10,000 images on CPU takes on the order of hours; on a modern GPU it takes minutes.
- **Increase `batch_size`** if your GPU has headroom. For a 12 GB card, try 64–128. Decrease it if you hit CUDA out-of-memory errors.
- **Re-indexing is incremental.** After the initial index build, subsequent runs only process new or changed files. Keeping the `db_path` on a fast local SSD improves search latency.
- **Keep the watcher off when batch-importing.** If you're dropping thousands of files at once, run `cli.py index` manually afterward rather than letting the watcher process each event individually.
- **EXR files are slower to load** than other formats due to format conversion overhead. If you have large EXR sequences, consider whether you need them indexed at all, or prefer to index exported JPEG/PNG derivatives instead.
- **Model loading takes ~2–5 seconds.** The CLI reloads the model on every invocation. If you're running many sequential searches, use the web UI instead — the server loads the model once and keeps it in memory.

---

## Roadmap

- **Duplicate and near-duplicate detection** — cluster embeddings by similarity to surface groups of visually redundant images across your library
- **Auto-tagging with BLIP-2** — generate descriptive captions for indexed images and store them alongside embeddings for hybrid keyword + semantic search
- **Collection filtering** — filter search results by source folder in the web UI without maintaining separate indexes
- **Image-to-image search** — drag and drop an image into the UI to find visually similar results using its embedding directly, bypassing the text query entirely
- **Metadata sidebar** — show EXIF/file metadata and similarity score in the result detail view
- **Export search results** — copy matched file paths to clipboard or export as a plaintext list for use in other tools
- **Configurable re-ranking** — optional cross-encoder reranking pass for higher-precision top results at the cost of some latency

---

## License

MIT
