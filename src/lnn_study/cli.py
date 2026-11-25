"""Small explicit commands; only `run` performs the complete selected pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .config import load_config, make_manifest
from .evaluation import evaluate_checkpoint, evaluate_references, make_probes
from .interpretability import interpret_checkpoint
from .io import read_json, write_json
from .training import train_one


def selected(spec, args):
    return all(getattr(args, key) is None or spec[key] == getattr(args, key) for key in ("family", "wiring", "task", "size", "seed"))


def checkpoint_paths(root, specs):
    for spec in specs:
        run_dir = root/"runs"/spec["run_id"]
        mapping = run_dir/"checkpoints.json"
        if mapping.exists():
            for filename in sorted(set(read_json(mapping).values()), key=lambda f: int(Path(f).stem.split("_")[-1])):
                yield run_dir/filename


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "train", "evaluate", "probe", "interpret", "analyze", "pareto", "run"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON manifest file for the read-only plan command")
    parser.add_argument("--family", choices=["cfc", "ltc", "gru", "lstm", "mlp", "rnn"])
    parser.add_argument("--wiring", choices=["dense", "ncp"])
    parser.add_argument("--task", choices=["balance", "swingup"])
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, help="Limit number of selected seed trajectories")
    parser.add_argument("--force", action="store_true", help="Recompute evaluation or interpretation artifacts")
    parser.add_argument("--stop-after-updates", type=int, help="Save an interruptible training checkpoint for a bounded run")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1 or args.stop_after_updates is not None and args.stop_after_updates < 1:
        parser.error("Limits must be positive")
    if args.config is not None:
        c = load_config(args.config)
    elif args.run_dir is not None and (args.run_dir/"resolved_config.json").exists():
        c = read_json(args.run_dir/"resolved_config.json")
    else:
        c = load_config("configs/smoke.yaml")
    root = args.run_dir or Path(c["output"])
    manifest = make_manifest(c)
    specs = [s for s in manifest["runs"] if selected(s, args)]
    if args.limit is not None:
        specs = specs[:args.limit]
    if not specs:
        parser.error("No runs match selection")
    if args.command == "plan":
        summary = {k: v for k, v in manifest.items() if k != "runs"}
        summary["selected_trajectories"] = len(specs)
        summary["selected_training_transitions"] = len(specs)*max(c["budgets"])
        summary["checkpoint_storage_estimate_note"] = "Exact bytes are logged after smoke checkpoints; optimizer and RNG snapshots add storage beyond actor weights."
        print(json.dumps(summary, indent=2))
        if args.output:
            write_json(args.output, manifest)
        return
    if args.command in ("train", "run"):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root/"manifest.json", manifest)
        write_json(root/"manifests"/(manifest["config_hash"]+".json"), manifest)
        write_json(root/"resolved_config.json", c)
        for i, spec in enumerate(specs):
            print(f"[{i+1}/{len(specs)}] train {spec['run_id']}", flush=True)
            print(json.dumps(train_one(spec, c, root, stop_after_updates=args.stop_after_updates)), flush=True)
    if args.command in ("evaluate", "run"):
        print("Evaluating physical references", flush=True)
        evaluate_references(c, root, force=args.force)
        for path in checkpoint_paths(root, specs):
            print(f"evaluate {path}", flush=True)
            evaluate_checkpoint(path, force=args.force)
    if args.command in ("probe", "run"):
        print(f"Shared probes: {make_probes(c, root, force=args.force)}", flush=True)
    if args.command in ("interpret", "run"):
        for path in checkpoint_paths(root, specs):
            print(f"interpret {path}", flush=True)
            interpret_checkpoint(path, root, force=args.force)
    if args.command in ("analyze", "pareto", "run"):
        print(json.dumps(analyze(root, c)), flush=True)


if __name__ == "__main__":
    main()
