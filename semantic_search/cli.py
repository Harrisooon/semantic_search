"""Command-line interface for semantic_search.

Commands
--------
semantic-search index [PATH ...]   Index folders (defaults to watched_folders in config)
semantic-search search "query"     Search with natural language
semantic-search status             Show index statistics
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_model_and_store(cfg: dict):
    """Load model + store. Deferred so 'status' never waits for model load."""
    from .models import ModelManager
    from .store import ImageStore

    model = ModelManager(cfg["model_variant"], cfg["device"])
    store = ImageStore(cfg["db_path"], embedding_dim=model.embedding_dim)
    return model, store


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace, cfg: dict) -> None:
    from .indexer import index_folders

    folders = args.paths or cfg["watched_folders"]
    if not folders:
        sys.exit(
            "No folders specified and watched_folders is empty in config.yaml."
        )

    model, store = _get_model_and_store(cfg)

    print(f"Indexing {len(folders)} folder(s) ...")
    stats = index_folders(
        folders=folders,
        store=store,
        model=model,
        supported_extensions=cfg["supported_extensions"],
        batch_size=cfg["batch_size"],
    )

    print(
        f"\nDone.\n"
        f"  {stats['total']:>6}  files found\n"
        f"  {stats['indexed']:>6}  indexed (new/changed)\n"
        f"  {stats['skipped']:>6}  skipped (unchanged)\n"
        f"  {stats['failed']:>6}  failed to load\n"
        f"  {stats['stale_removed']:>6}  stale records removed"
    )


def cmd_search(args: argparse.Namespace, cfg: dict) -> None:
    from .search import search

    model, store = _get_model_and_store(cfg)

    if store.count() == 0:
        sys.exit("Index is empty. Run `semantic-search index` first.")

    results = search(args.query, store, model, top_k=args.top)

    if not results:
        print("No results.")
        return

    for path, score in results:
        print(f"{score:.4f}  {path}")


def cmd_status(args: argparse.Namespace, cfg: dict) -> None:
    from .store import ImageStore

    # Status never needs the model — open the store directly with the default dim.
    # The dim is only used when creating a new table; for reads it's irrelevant.
    store = ImageStore(cfg["db_path"])
    stats = store.stats()

    # Index summary
    print(f"Index path  : {cfg['db_path']}")
    print(f"Total files : {stats['total']}")

    if stats.get("last_indexed"):
        ts = datetime.datetime.fromtimestamp(stats["last_indexed"])
        print(f"Last indexed: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Last indexed: never")

    # Watched folders
    print(f"\nWatched folders ({len(cfg['watched_folders'])}):")
    for folder in cfg["watched_folders"]:
        tag = "[ok]     " if Path(folder).exists() else "[missing]"
        print(f"  {tag}  {folder}")

    # Per-folder breakdown (only shown when there's something indexed)
    if stats["total"] > 0:
        print(f"\nBreakdown by folder:")
        for folder, count in sorted(
            stats["by_folder"].items(), key=lambda x: -x[1]
        ):
            print(f"  {count:>6}  {folder}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-search",
        description="Local semantic image search for CG artists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  semantic-search index ~/references\n'
            '  semantic-search index                       # uses watched_folders in config\n'
            '  semantic-search search "moody forest lighting"\n'
            '  semantic-search search "sci-fi panel detail" --top 10\n'
            "  semantic-search status\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- index ---------------------------------------------------------------
    p_index = sub.add_parser(
        "index",
        help="Index folders of images",
        description=(
            "Walk folder(s), compute blake2b hashes, embed changed files with "
            "SigLIP 2, and store the results in LanceDB. Unchanged files are "
            "skipped. If no PATH is given, watched_folders from config is used."
        ),
    )
    p_index.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="Folder(s) to index. Overrides watched_folders if given.",
    )

    # -- search --------------------------------------------------------------
    p_search = sub.add_parser(
        "search",
        help="Search the index with a text query",
        description=(
            "Encode the query with SigLIP 2 and return the most similar images "
            "from the index, ranked by cosine similarity."
        ),
    )
    p_search.add_argument(
        "query",
        help='Natural language query, e.g. "warm backlit portrait"',
    )
    p_search.add_argument(
        "--top", "-n",
        type=int,
        default=20,
        metavar="N",
        help="Number of results to return (default: 20)",
    )

    # -- status --------------------------------------------------------------
    sub.add_parser(
        "status",
        help="Show index statistics",
        description="Print total indexed files, last-indexed timestamp, and per-folder counts.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    from . import load_config

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

    dispatch = {
        "index": cmd_index,
        "search": cmd_search,
        "status": cmd_status,
    }
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
