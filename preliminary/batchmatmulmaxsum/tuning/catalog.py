"""Load and validate the candidate catalog used to generate C++ definitions."""

from __future__ import annotations

import json
from pathlib import Path

from .model import TilingConfig


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "configs" / "tiling_candidates.json"


def load_catalog(path: Path = DEFAULT_CATALOG) -> list[TilingConfig]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported candidate catalog schema")
    configs = [TilingConfig(**item) for item in document.get("candidates", [])]
    if not configs:
        raise ValueError("candidate catalog is empty")

    keys = [item.key for item in configs]
    symbols = [item.symbol for item in configs]
    names = [item.name for item in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate keys must be unique")
    if len(symbols) != len(set(symbols)):
        raise ValueError("candidate symbols must be unique")
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")

    for item in configs:
        if item.path not in {"reference", "auto_fused"}:
            raise ValueError(f"unsupported path for key {item.key}: {item.path}")
        if min(item.tile_m, item.tile_n, item.tile_k, item.vec_m, item.vec_n) <= 0:
            raise ValueError(f"tile sizes must be positive for key {item.key}")
        if item.split_n <= 0:
            raise ValueError(f"split_n must be positive for key {item.key}")
        if item.split_n != 1:
            raise ValueError(
                f"split_n is reserved but not implemented by key {item.key}"
            )
    return configs
