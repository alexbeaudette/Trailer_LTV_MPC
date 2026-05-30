"""Load YAML controller configuration into typed config dataclasses."""

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import TrailerLtvMpcConfig
from .geometry import Geometry


_MATRIX_FIELDS = {"Q", "Qf", "Q_rev", "Qf_rev", "Q_fwd", "Qf_fwd", "R", "Rd"}
_LOADER_KEYS = {"include"}


def load_controller_config(path) -> TrailerLtvMpcConfig:
    """Load a YAML config stack and return a TrailerLtvMpcConfig."""
    config_path = Path(path)
    data = _load_config_stack(config_path, seen=set())
    return _build_config(data)


def _load_config_stack(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Config include cycle detected at {path}.")
    seen.add(resolved)

    data = _read_yaml_mapping(resolved)
    merged: dict[str, Any] = {}
    include_entries = data.get("include", [])
    if isinstance(include_entries, (str, Path)):
        include_entries = [include_entries]
    if not isinstance(include_entries, list):
        raise ValueError("Config include must be a path string or list of path strings.")

    for include_path in include_entries:
        child_path = resolved.parent / str(include_path)
        merged = _deep_merge(merged, _load_config_stack(child_path, seen))

    own_data = {key: value for key, value in data.items() if key not in _LOADER_KEYS}
    return _deep_merge(merged, own_data)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(data: dict[str, Any]) -> TrailerLtvMpcConfig:
    config_fields = {field.name for field in fields(TrailerLtvMpcConfig)}
    unknown = sorted(set(data) - config_fields)
    if unknown:
        raise ValueError(f"Unknown controller config key(s): {', '.join(unknown)}.")

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key == "geom":
            kwargs[key] = _build_geometry(value)
        elif key in _MATRIX_FIELDS:
            kwargs[key] = np.asarray(value, dtype=float)
        else:
            kwargs[key] = value
    return TrailerLtvMpcConfig(**kwargs)


def _build_geometry(data: Any) -> Geometry:
    if data is None:
        return Geometry()
    if not isinstance(data, dict):
        raise ValueError("geom must be a YAML mapping.")
    geometry_fields = {field.name for field in fields(Geometry)}
    unknown = sorted(set(data) - geometry_fields)
    if unknown:
        raise ValueError(f"Unknown geometry config key(s): {', '.join(unknown)}.")
    return Geometry(**data)
