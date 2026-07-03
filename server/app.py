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
import struct
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from semantic_search import load_config
from semantic_search.store import ImageStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Startup — load config and store now; load the model in a background thread
# so the server (and UI) come up immediately. Only /search and reindexing
# need the model; browse/folders/thumbs/similar all work from the store.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE.parent / "config.yaml"

cfg = load_config(_CONFIG_PATH)
store = ImageStore(cfg["db_path"])

model = None  # set by _load_model_background once ready
_model_error: str | None = None
_model_ready = threading.Event()


def _load_model_background():
    """Import torch/transformers and load the model off the critical path."""
    global model, _model_error
    try:
        from semantic_search.models import ModelManager

        m = ModelManager(cfg["model_variant"], cfg["device"])
        # On a fresh install the table is created lazily; make sure it gets
        # the real embedding dim rather than the ImageStore default.
        store.embedding_dim = m.embedding_dim
        model = m
        logger.info("Model ready — semantic search enabled")
    except Exception as e:
        _model_error = f"{type(e).__name__}: {e}"
        logger.error("Model failed to load: %s", e)
    finally:
        _model_ready.set()


threading.Thread(target=_load_model_background, daemon=True).start()

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
        self.progress = 0
        self.total = 0
        self.last_result: dict | None = None
        self.last_run: float | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Returns False if already running."""
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.progress = 0
            self.total = 0
            return True

    def set_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.progress = done
            self.total = total

    def finish(self, result: dict):
        with self._lock:
            self.running = False
            self.last_result = result
            self.last_run = time.time()

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "progress": self.progress,
                "total": self.total,
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


@app.get("/folders")
async def folders_endpoint():
    return {"folders": store.get_all_folders()}


@app.get("/browse")
async def browse_endpoint(
    folder: list[str] = Query(default=[]),
    color: list[str] = Query(default=[]),
    top: int = Query(default=60, ge=1, le=200),
):
    """Filter-only scan — no semantic query needed."""
    if store.count() == 0:
        return {"results": [], "error": "Index is empty."}
    results = store.browse(
        folder_filter=folder or None,
        color_filter=color or None,
        top_k=top,
    )
    return {"results": results}


@app.get("/search")
async def search_endpoint(
    q: str = Query(min_length=1, max_length=500),
    top: int = Query(default=40, ge=1, le=200),
    folder: list[str] = Query(default=[]),
    color: list[str] = Query(default=[]),
):
    if model is None:
        if _model_error:
            return {"results": [], "error": f"Model failed to load: {_model_error}"}
        return {"results": [], "loading": True}

    if store.count() == 0:
        return {"results": [], "error": "Index is empty. Run `semantic-search index` first."}

    from semantic_search.search import search as _search

    results = _search(
        q, store, model, top_k=top,
        folder_filter=folder or None,
        color_filter=color or None,
    )
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

        from semantic_search.indexer import load_image

        img = load_image(file_path)
        if img is None:
            raise HTTPException(status_code=422, detail="Could not decode image")
        img.thumbnail((size, size), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
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
    folder: list[str] = Query(default=[]),
    color: list[str] = Query(default=[]),
):
    if store.count() == 0:
        return {"results": [], "error": "Index is empty."}

    embedding = store.get_embedding(path)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Image not in index")

    posix_path = Path(path).as_posix()
    results = store.search(
        embedding, top_k=top + 1,
        folder_filter=folder or None,
        color_filter=color or None,
    )
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
# Copy-file-to-clipboard endpoint
#
# Puts the actual file on the OS clipboard (Windows CF_HDROP — the same
# format Explorer uses for Ctrl+C), so pasting into Explorer/Discord/Slack
# pastes the real file, not a re-encoded thumbnail. Animated GIFs stay
# animated. Browser JS cannot do this, but the server runs locally.
# ---------------------------------------------------------------------------

def _set_clipboard_files(paths: list[Path]) -> None:
    if sys.platform != "win32":
        raise NotImplementedError("Copy file is only supported on Windows")

    import ctypes
    from ctypes import wintypes

    CF_HDROP = 15
    GMEM_MOVEABLE = 0x0002

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    # Explicit signatures — the ctypes defaults truncate 64-bit handles.
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]

    # DROPFILES header (20 bytes: file-list offset, drop point, fNC, fWide=1
    # for UTF-16) followed by a double-NUL-terminated UTF-16 path list.
    file_list = "\0".join(str(p) for p in paths) + "\0\0"
    payload = struct.pack("<IiiII", 20, 0, 0, 0, 1) + file_list.encode("utf-16-le")

    hglobal = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(payload))
    if not hglobal:
        raise RuntimeError("GlobalAlloc failed")
    ptr = kernel32.GlobalLock(hglobal)
    if not ptr:
        kernel32.GlobalFree(hglobal)
        raise RuntimeError("GlobalLock failed")
    ctypes.memmove(ptr, payload, len(payload))
    kernel32.GlobalUnlock(hglobal)

    # Another app may briefly hold the clipboard — retry a few times.
    for attempt in range(10):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        kernel32.GlobalFree(hglobal)
        raise RuntimeError("Could not open clipboard (held by another application)")

    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_HDROP, hglobal):
            kernel32.GlobalFree(hglobal)
            raise RuntimeError("SetClipboardData failed")
        # On success the system owns the memory — do not free it.
    finally:
        user32.CloseClipboard()


@app.post("/copy_file")
async def copy_file_endpoint(path: str, request: Request):
    # Require a custom header so cross-origin pages can't trigger this —
    # it forces a CORS preflight, which fails for other origins.
    if request.headers.get("x-requested-with") != "semantic_search":
        raise HTTPException(status_code=403, detail="Bad request origin")

    file_path = Path(path)
    if not _is_allowed(file_path):
        raise HTTPException(status_code=403, detail="Path not in a watched folder")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        _set_clipboard_files([file_path.resolve()])
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error("Clipboard copy failed for %s: %s", path, e)
        raise HTTPException(status_code=500, detail=f"Clipboard copy failed: {e}")
    return {"copied": True}


# ---------------------------------------------------------------------------
# Search-by-image endpoint (drag-and-drop upload — the image does not need
# to be in the index)
# ---------------------------------------------------------------------------

@app.post("/search_by_image")
async def search_by_image(
    file: UploadFile = File(...),
    top: int = Query(default=40, ge=1, le=200),
    folder: list[str] = Query(default=[]),
    color: list[str] = Query(default=[]),
):
    if model is None:
        if _model_error:
            return {"results": [], "error": f"Model failed to load: {_model_error}"}
        return {"results": [], "loading": True}

    if store.count() == 0:
        return {"results": [], "error": "Index is empty. Run `semantic-search index` first."}

    data = await file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (100 MB max)")

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image")

    embedding = model.encode_image([img])[0]
    results = store.search(
        embedding, top_k=top,
        folder_filter=folder or None,
        color_filter=color or None,
    )
    return {
        "results": [
            {"path": p, "filename": Path(p).name, "score": s}
            for p, s in results
        ],
        "query_filename": file.filename,
    }


# ---------------------------------------------------------------------------
# Reindex endpoints
# ---------------------------------------------------------------------------

def _run_reindex():
    from semantic_search.indexer import index_folders
    result = None
    try:
        _model_ready.wait()
        if model is None:
            raise RuntimeError(f"Model failed to load: {_model_error}")
        result = index_folders(
            folders=cfg["watched_folders"],
            store=store,
            model=model,
            supported_extensions=cfg["supported_extensions"],
            batch_size=cfg["batch_size"],
            progress_callback=_job.set_progress,
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
