"""Validated configuration, stable RNG streams, and grid manifests."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import yaml

FAMILIES = ("cfc", "ltc", "gru", "lstm", "mlp", "rnn")
TASKS = ("balance", "swingup")
DEFAULTS = {
    "label": "scientific", "output": "results/study", "seed": 2026,
    "families": list(FAMILIES), "wirings": ["dense", "ncp"],
    "sizes": [8, 12, 16, 24, 32, 64], "budgets": [100000, 300000, 1000000, 2000000, 5000000],
    "seeds": list(range(10)), "tasks": list(TASKS), "sparsities": [0.5],
    "model": {"motor_neurons": 4, "ode_unfolds": 6, "critic_width": 64},
    "env": {"horizon": 500, "dt": 0.02, "force_limit": 10., "disturbance": 5., "pulse_duration": 3},
    "ppo": {"n_envs": 16, "rollout_steps": 128, "sequence_length": 16,
            "minibatch_sequences": 16, "epochs": 4, "learning_rate": 0.001,
            "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "value_coef": 0.5,
            "entropy_coef": 0.005, "max_grad_norm": 0.5, "threads": 1},
    "family_overrides": {},
    "evaluation": {"episodes": 100, "dwell": 50, "final_window": 200, "failure_cost": 1000.,
                   "recovery_dwell": 10},
    "probe": {"episodes": 48, "burn_in": 8, "bins": 10, "tree_depth": 3,
              "min_leaf_fraction": 0.1, "ccp_alpha": 0.01, "activity_epsilon": 1e-6},
    "analysis": {"bootstrap": 200, "survival_min": 0.95, "swingup_min": 0.8, "balance_cost_ratio_max": 2.,
                 "balanced_min": 0.8, "tree_fidelity_min": 0.5, "active_min": 0.8,
                 "match_relative_cost": 0.1, "match_success": 0.05,
                 "budget_axes": ["actual_steps", "training_seconds"]},
}


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_for(base, *coordinates):
    return int(digest([int(base), *coordinates])[:8], 16)


def load_config(path):
    path = Path(path).resolve()
    def read(p, seen):
        if p in seen:
            raise ValueError("Cyclic configuration inheritance")
        data = yaml.safe_load(p.read_text()) or {}
        parent = data.pop("extends", None)
        return merge(read((p.parent / parent).resolve(), seen | {p}) if parent else {}, data)
    config = merge(DEFAULTS, read(path, set()))
    validate(config)
    return config


def validate(c):
    unknown = set(c) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    for name in ("families", "wirings", "sizes", "budgets", "seeds", "tasks", "sparsities"):
        if not c[name] or len(c[name]) != len(set(c[name])):
            raise ValueError(f"{name} must be a nonempty list without duplicates")
    if set(c["families"]) - set(FAMILIES) or set(c["tasks"]) - set(TASKS) or set(c["wirings"]) - {"dense", "ncp"}:
        raise ValueError("Unknown model, wiring or task")
    if c["label"] != "smoke" and len(c["seeds"]) < 6:
        raise ValueError("Scientific cells require at least six independent seeds")
    if any(not isinstance(s, int) or s < 0 for s in c["seeds"]):
        raise ValueError("Replicate seeds must be nonnegative integers")
    if any(not isinstance(b, int) for b in c["budgets"]+c["sizes"]) or c["budgets"] != sorted(c["budgets"]) or min(c["budgets"]) <= 0:
        raise ValueError("Budgets must be positive and increasing")
    if min(c["sizes"]) < 1 or ("ncp" in c["wirings"] and min(c["sizes"]) <= c["model"]["motor_neurons"] + 2):
        raise ValueError("NCP sizes must exceed motor_neurons + 2")
    if any(not 0 <= v <= 0.9 for v in c["sparsities"]):
        raise ValueError("NCPS sparsities must be in [0, 0.9]")
    for ppo in [c["ppo"], *[merge(c["ppo"], v) for v in c["family_overrides"].values()]]:
        for key in ("n_envs", "rollout_steps", "sequence_length", "minibatch_sequences", "epochs", "threads"):
            if not isinstance(ppo[key], int) or ppo[key] < 1:
                raise ValueError(f"PPO {key} must be positive")
    if c["env"]["horizon"] < 20 or c["env"]["dt"] <= 0 or c["env"]["force_limit"] <= 0:
        raise ValueError("Invalid environment settings")
    if c["evaluation"]["episodes"] < 3 or c["probe"]["episodes"] < 6:
        raise ValueError("Need >=3 evaluation episodes and >=6 probe trajectories for held-out splits")
    if c["probe"]["burn_in"] >= c["env"]["horizon"] or c["probe"]["bins"] < 2:
        raise ValueError("Invalid probe burn-in or bins")
    if c["analysis"]["bootstrap"] < 1:
        raise ValueError("Need positive bootstrap count")
    if set(c["family_overrides"]) - set(FAMILIES):
        raise ValueError("Unknown family override")
    if not c["analysis"]["budget_axes"] or set(c["analysis"]["budget_axes"]) - {"actual_steps", "training_seconds"}:
        raise ValueError("Unknown Pareto budget axis")


def make_manifest(c):
    runs = []
    for task, family, wiring, size, replicate in itertools.product(c["tasks"], c["families"], c["wirings"], c["sizes"], c["seeds"]):
        for sparsity in c["sparsities"] if wiring == "ncp" else [None]:
            spec = dict(task=task, family=family, wiring=wiring, size=size, seed=replicate, sparsity=sparsity)
            spec["topology_seed"] = seed_for(c["seed"], "topology", task, size, replicate)
            identity = {"spec": spec, "model": c["model"], "env": c["env"],
                        "ppo": merge(c["ppo"], c["family_overrides"].get(family, {})),
                        "base_seed": c["seed"], "label": c["label"]}
            spec["run_id"] = f"{task}-{family}-{wiring}-h{size}-s{replicate}-{digest(identity)[:10]}"
            runs.append(spec)
    cells = len(runs) // len(c["seeds"]) * len(c["budgets"])
    return {"schema_version": 1, "config_hash": digest(c), "label": c["label"], "runs": runs,
            "scientific_cells": cells, "seed_trajectories": len(runs),
            "checkpoint_records": len(runs) * len(c["budgets"]),
            "training_transitions_requested": len(runs) * max(c["budgets"]),
            "note": "Evaluation/probes excluded; update-boundary overshoot is recorded. Budgets share training prefixes."}
