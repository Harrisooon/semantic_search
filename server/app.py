"""FastAPI web server.

Endpoints:
  GET  /               — search UI
  GET  /search?q=...   — JSON search results
  GET  /thumb?path=... — on-the-fly JPEG thumbnail
  POST /reindex        — trigger incremental reindex in background
  GET  /reindex/status — poll reindex job status
"""

from __future__ import annotations

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
model = ModelManager(cfg["model_variant"], cfg["device"])
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


@app.get("/search")
async def search_endpoint(
    q: str = Query(min_length=1),
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
            headers={"Cache-Control": "max-age=86400"},
        )
    except Exception as e:
        logger.error("Failed to serve thumb for %s: %s", path, e)
        raise HTTPException(status_code=500, detail="Could not load image")


# ---------------------------------------------------------------------------
# Reindex endpoints
# ---------------------------------------------------------------------------

def _run_reindex():
    from semantic_search.indexer import index_folders
    try:
        result = index_folders(
            folders=cfg["watched_folders"],
            store=store,
            model=model,
            supported_extensions=cfg["supported_extensions"],
            batch_size=cfg["batch_size"],
        )
        _job.finish(result)
        logger.info("Reindex complete: %s", result)
    except Exception as e:
        logger.error("Reindex failed: %s", e)
        _job.finish({"error": str(e)})


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
    print(f"Starting server — open http://localhost:8000 in your browser")
    print(f"Index: {store.count()} images")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
