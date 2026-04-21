"""Tiny JSON-backed registry mapping (dataset, curvature, rewiring) -> wandb sweep id."""

import json
import os
import tempfile
from typing import Optional

DEFAULT_REGISTRY_PATH = os.path.join("config", "sweep_registry.json")


def _key(dataset: str, curvature: str, rewiring: bool) -> str:
    return f"{dataset}|{curvature}|{bool(rewiring)}"


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sweep_registry.", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def lookup(dataset: str, curvature: str, rewiring: bool,
           path: str = DEFAULT_REGISTRY_PATH) -> Optional[str]:
    return _load(path).get(_key(dataset, curvature, rewiring))


def record(dataset: str, curvature: str, rewiring: bool, sweep_id: str,
           path: str = DEFAULT_REGISTRY_PATH) -> None:
    data = _load(path)
    data[_key(dataset, curvature, rewiring)] = sweep_id
    _save_atomic(path, data)
