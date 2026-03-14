"""FastAPI web server.

Endpoints:
  GET  /               — search UI
  GET  /search?q=...   — JSON search results
  GET  /thumb?path=... — on-the-fly JPEG thumbnail
  POST /reindex        — trigger incremental reindex in background
  GET  /reindex/status — poll reindex job status
"""

from __future__ import annotations

print("Starting up...")

import io
import logging
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from semantic_search import load_config
from semantic_search.models import ModelManager
from semantic_search.search import search as _search
from semantic_search.store import ImageStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Startup — load config, model, store once
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE.parent / "config.yaml"

cfg = load_config(_CONFIG_PATH)
print("Loading model...")
model = ModelManager(cfg["model_variant"], cfg["device"])
print("Opening index...")
store = ImageStore(cfg["db_path"], embedding_dim=model.embedding_dim)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="semantic_search", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=_HERE / "templates")

# ---------------------------------------------------------------------------
# Reindex job state
# ---------------------------------------------------------------------------

class _ReindexJob:
    def __init__(self):
        self.running = False
        self.last_result: dict | None = None
        self.last_run: float | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Returns False if already running."""
        with self._lock:
            if self.running:
                return False
            self.running = True
            return True

    def finish(self, result: dict):
        with self._lock:
            self.running = False
            self.last_result = result
            self.last_run = time.time()

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "last_result": self.last_result,
                "last_run": self.last_run,
            }

_job = _ReindexJob()


# ---------------------------------------------------------------------------
# Security helper
# ---------------------------------------------------------------------------

def _is_allowed(path: Path) -> bool:
    """Only serve files that live inside one of the configured watched folders."""
    resolved = path.resolve()
    for folder in cfg["watched_folders"]:
        try:
            resolved.relative_to(Path(folder).resolve())
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/count")
async def count():
    return {"count": store.count()}


@app.get("/search")
async def search_endpoint(
    q: str = Query(min_length=1, max_length=500),
    top: int = Query(default=40, ge=1, le=200),
):
    if store.count() == 0:
        return {"results": [], "error": "Index is empty. Run `semantic-search index` first."}

    results = _search(q, store, model, top_k=top)
    return {
        "results": [
            {"path": path, "filename": Path(path).name, "score": score}
            for path, score in results
        ]
    }


@app.get("/thumb")
async def thumb(
    path: str,
    size: int = Query(default=400, ge=32, le=3840),
):
    file_path = Path(path)

    if not _is_allowed(file_path):
        raise HTTPException(status_code=403, detail="Path not in a watched folder")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        from PIL import Image

        img = Image.open(file_path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as e:
        logger.error("Failed to serve thumb for %s: %s", path, e)
        raise HTTPException(status_code=500, detail="Could not load image")


# ---------------------------------------------------------------------------
# Raw file endpoint (used for animated GIFs so animation is preserved)
# ---------------------------------------------------------------------------

@app.get("/raw")
async def raw(path: str):
    file_path = Path(path)

    if not _is_allowed(file_path):
        raise HTTPException(status_code=403, detail="Path not in a watched folder")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    suffix = file_path.suffix.lower()
    media_types = {
        ".gif": "image/gif",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tga": "image/x-tga",
        ".exr": "image/x-exr",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return Response(
        content=file_path.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Similar images endpoint
# ---------------------------------------------------------------------------

@app.get("/similar")
async def similar_endpoint(
    path: str,
    top: int = Query(default=40, ge=1, le=200),
):
    if store.count() == 0:
        return {"results": [], "error": "Index is empty."}

    embedding = store.get_embedding(path)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Image not in index")

    posix_path = Path(path).as_posix()
    results = store.search(embedding, top_k=top + 1)
    # Exclude the query image itself (it will be the top hit with score ~1.0).
    # store.search returns posix paths; compare against normalised posix_path.
    results = [(p, s) for p, s in results if p != posix_path][:top]

    return {
        "results": [
            {"path": p, "filename": Path(p).name, "score": s}
            for p, s in results
        ],
        "query_path": path,
        "query_filename": Path(path).name,
    }


# ---------------------------------------------------------------------------
# Reindex endpoints
# ---------------------------------------------------------------------------

def _run_reindex():
    from semantic_search.indexer import index_folders
    result = None
    try:
        result = index_folders(
            folders=cfg["watched_folders"],
            store=store,
            model=model,
            supported_extensions=cfg["supported_extensions"],
            batch_size=cfg["batch_size"],
        )
        logger.info("Reindex complete: %s", result)
    except Exception as e:
        logger.error("Reindex failed: %s", e)
        result = {"error": str(e)}
    finally:
        _job.finish(result or {"error": "Reindex did not complete"})


@app.post("/reindex")
async def reindex():
    if not _job.start():
        return {"started": False, "message": "Reindex already in progress"}
    thread = threading.Thread(target=_run_reindex, daemon=True)
    thread.start()
    return {"started": True}


@app.get("/reindex/status")
async def reindex_status():
    return _job.status()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser
    url = "http://localhost:8000"
    print(f"Starting server — {url}")
    print(f"Index: {store.count()} images")
    # Open browser after a short delay so the server is ready to accept connections.
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
