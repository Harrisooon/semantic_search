"""semantic_search — local semantic image search for CG artists."""

__version__ = "0.1.0"

from pathlib import Path

import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load and normalise config.yaml.

    Resolves ~ in paths and returns supported_extensions as a lowercase set.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    cfg["db_path"] = str(Path(cfg["db_path"]).expanduser())
    cfg["watched_folders"] = [
        str(Path(p).expanduser()) for p in cfg.get("watched_folders", [])
    ]
    cfg["supported_extensions"] = {
        ext.lower() for ext in cfg.get("supported_extensions", [])
    }
    return cfg
