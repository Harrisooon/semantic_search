# Handover Notes

## What this project is

A fully local semantic image search tool for CG artists. Images are embedded using
SigLIP 2 (vision-language model), stored in LanceDB, and searched via a FastAPI web
UI. No cloud services — everything runs on the user's machine.

## Architecture

```
config.yaml                   ← user configuration (watched folders, db path, model)
semantic_search/
  __init__.py                 ← load_config(), normalises paths and extensions
  models.py                   ← ModelManager: loads SigLIP 2, encode_image/encode_text
  store.py                    ← ImageStore: LanceDB wrapper (upsert, search, hashes)
  indexer.py                  ← index_folders(): walk → hash → diff → embed → upsert
  search.py                   ← search(): text query → ranked results
  cli.py                      ← `semantic-search index/search/status` CLI commands
server/
  app.py                      ← FastAPI server (search, thumb, similar, reindex)
  templates/index.html        ← Single-page dark UI (search, lightbox, find-similar)
launch.bat                    ← Double-click to start server + open browser
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

## Known gotchas / bugs fixed

### 1. Backslash paths in JS onclick attributes
`search.py` converts posix paths to Windows paths (`D:\references\foo.jpg`).
When these are embedded in HTML `onclick="..."` attributes, `\r` in `\references`
is interpreted by the JS parser as a carriage return, mangling the string.

**Fix applied:** `escAttr()` in `index.html` now escapes backslashes first:
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

## Current state

- Text search: working ✓
- Incremental reindex (button in UI): working ✓ — skips unchanged files by hash
- Find similar (hover card → "find similar" button): recently fixed, should work
  after server restart with latest code
- Lightbox (click image): working ✓
- Copy path (hover card → "copy path"): working ✓

## Environment

- Windows 10, Python (no venv — installed to system Python)
- PyTorch installed for CUDA (`cu118`): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Package installed editable: `pip install -e .`
- `config.yaml` at repo root — user has set `watched_folders: [D:/references]`
- DB at `~/.semantic_search/db` (LanceDB directory)

## Running the server

```
python server/app.py
```
Or double-click `launch.bat`. Server starts at http://localhost:8000 and auto-opens browser.

## Running the CLI indexer

```
semantic-search index        # index/reindex watched folders
semantic-search status       # show count and last-indexed time (no model load)
semantic-search search "query"
```

## Git history (recent)

- `36a5ef2` Fix incremental reindex: replace broken lance scanner with to_arrow()
- `a2baaa6` re-index functionality
- `07aba86` local instance interface
- `c0968ed` fixed tensor model, updated watched folders

## Pending / to-do

- The "find similar" feature is new and was just debugged. Verify it works end-to-end
  after the latest server restart.
- Consider whether `search.py` should stop converting to OS-native paths. Returning
  posix paths everywhere would eliminate the backslash JS-escaping problem entirely
  and make `/search` and `/similar` consistent. Trade-off: copy-path in the UI would
  give forward-slash paths to the user (fine for most uses on Windows).
- `get_embedding()` loads the entire table into memory to filter by path. For large
  libraries (10k+ images) this will be slow. A proper indexed lookup or LanceDB
  filter query would be more efficient.
