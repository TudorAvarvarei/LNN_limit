"""Atomic artifacts and environment provenance."""
import importlib.metadata
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def provenance():
    packages = ["torch", "ncps", "gymnasium", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "PyYAML"]
    source = Path(__file__).parent
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(source.glob("*.py")))).hexdigest()
    return {"python": platform.python_version(), "platform": platform.platform(), "source_hash": source_hash,
            "processor": platform.processor(), "device": "cpu", "torch_threads": torch.get_num_threads(),
            "packages": {p: importlib.metadata.version(p) for p in packages}}
