# Handover Notes

## What this project is

A fully local semantic image search tool for CG artists. Images are embedded using
SigLIP 2 (vision-language model), stored in LanceDB, and searched via a FastAPI web
UI. No cloud services — everything runs on the user's machine.

## Architecture

```
config.yaml                   ← user configuration (watched folders, db path, model)
install.bat                   ← creates venv, installs CUDA PyTorch + dependencies
launch.bat                    ← activates venv, starts server, opens browser
semantic_search/
  __init__.py                 ← load_config(), normalises paths and extensions
  models.py                   ← ModelManager: loads SigLIP 2, encode_image/encode_text
  store.py                    ← ImageStore: LanceDB wrapper (upsert, search, hashes)
  indexer.py                  ← index_folders(): walk → hash → diff → embed → upsert
  search.py                   ← search(): text query → ranked results
  cli.py                      ← `semantic-search index/search/status` CLI commands
server/
  app.py                      ← FastAPI server (search, count, thumb, raw, similar, reindex)
  templates/index.html        ← Single-page dark UI (search, lightbox, find-similar, GIF support)
```

## Key design decisions

- **Paths stored as posix** (forward slashes) in LanceDB for cross-platform consistency.
  `indexer.py` stores `file_path.as_posix()`. All DB reads/writes use posix.
- **`search.py` converts posix → OS-native** before returning results to callers,
  so the CLI shows Windows backslash paths. The web server inherits this.
- **Incremental reindex** uses blake2b hashes. `get_all_hashes()` reads the stored
  hash per path; only files whose hash has changed are re-embedded.
- **`_read_columns()` in store.py** uses `to_arrow().select(columns)` — NOT the
  lance scanner API (`to_lance().scanner()`), which is broken in the installed lance
  version (`module 'lance' has no attribute 'dataset'`).
- **Model cache** — `models.py` tries `local_files_only=True` first to avoid HuggingFace
  HTTP freshness checks on every startup. Falls back to downloading on first run.
- **venv** — `install.bat` creates `.\venv\` and `launch.bat` activates it. CUDA 12.1
  PyTorch is installed first (before `pip install -e .`) to avoid pip overwriting it
  with the CPU build.

## Known gotchas / bugs fixed

### 1. Backslash paths in JS onclick attributes
`search.py` converts posix paths to Windows paths (`D:\references\foo.jpg`).
When these are embedded in HTML `onclick="..."` attributes, `\r` in `\references`
is interpreted by the JS parser as a carriage return, mangling the string.

**Fix applied:** `escAttr()` in `index.html` escapes backslashes first:
```js
function escAttr(s) {
  return s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
```

### 2. get_embedding() needs posix normalisation
`get_embedding(path)` receives an OS-native path from the web UI.
It must convert to posix before looking up in LanceDB.
`store.py` does `Path(path).as_posix()` before the arrow filter.

### 3. /similar self-exclusion uses posix path
`app.py` normalises the query path to posix before filtering out the query image
from results (since `store.search()` returns posix paths).

### 4. Reindex job state leak
`_run_reindex()` in `app.py` uses a `finally` block to guarantee `_job.finish()`
is always called, even if an unexpected exception fires before the result is set.
Without this, the reindex button would lock up permanently.

### 5. Thumbnail cache
`/thumb` and `/raw` endpoints use `Cache-Control: no-cache` (not `max-age=86400`)
so that reindexed or replaced files are always fetched fresh from disk.

## Current state (as of 2026-03-13)

All core features working:

- Text search ✓
- Find similar (visual embedding search) ✓
- Lightbox with copy path + find similar buttons ✓
- Arrow key navigation in lightbox (left/right) ✓
- Incremental reindex via UI button ✓ — auto-refreshes results on completion
- Reindex status shows: new / unchanged / failed / removed counts ✓
- GIF support ✓ — static thumbnail in grid, animates on hover, animated in lightbox
- GIF badge overlay on cards ✓
- Duplicate filtering by file hash (search + find similar) ✓
- Index count displayed in header, updated after reindex ✓
- Stale entry cleanup (deleted files pruned on reindex) ✓
- venv-based install via install.bat ✓

## Environment

- Windows 10, Python 3.10+
- Virtual environment at `.\venv\` (created by `install.bat`)
- PyTorch with CUDA 12.1: installed via `install.bat` from `download.pytorch.org/whl/cu121`
- Package installed editable: `pip install -e ".[server]"`
- `config.yaml` at repo root — user sets `watched_folders`
- DB at `~/.semantic_search/db` (LanceDB directory)

## Running the server

Double-click `launch.bat` (activates venv, starts server, opens browser), or manually:
```
venv\Scripts\activate
python server/app.py
```
Server starts at http://localhost:8000.

## Running the CLI indexer

```
semantic-search index        # index/reindex watched folders
semantic-search status       # show count and last-indexed time (no model load)
semantic-search search "query"
```

## Pending / worth revisiting

- `get_embedding()` loads the entire table into memory to filter by path. For very
  large libraries (50k+ images) this could be slow. A proper LanceDB filter query
  would be more efficient, but the current arrow-based approach is reliable.
- `search.py` converting posix → OS-native paths means the web UI shows Windows
  backslash paths in copy-path. Returning posix everywhere would simplify the JS
  escaping but would change the format users see.
