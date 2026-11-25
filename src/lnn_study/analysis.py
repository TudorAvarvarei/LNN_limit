"""Seed-level uncertainty, matched comparisons, and multiobjective frontiers."""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import make_manifest, seed_for
from .evaluation import evaluate_references
from .io import read_json, write_json

KEYS = ["task", "family", "wiring", "size", "sparsity", "requested_budget"]
METRICS = ["J_safe", "cost_ratio", "success", "balanced_final", "survival", "recovery_seconds", "return",
           "clean_J_safe", "clean_success", "clean_survival", "disturbance_cost_increase",
           "force_rms", "saturation", "chattering", "capture_steps", "training_seconds", "actual_steps",
           "actor_effective", "actor_stored", "core_unique_edges", "shared_MIG", "shared_modularity",
           "shared_tree_r2", "shared_tree_leaves", "shared_active_fraction", "on_policy_MIG", "on_policy_modularity", "on_policy_tree_r2"]


def nondominated(values):
    """All objectives minimized; invalid points never enter a frontier."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values).all(axis=1)
    result = np.zeros(len(values), dtype=bool)
    v = values[valid]
    dominates = (v[None, :, :] <= v[:, None, :]).all(-1) & (v[None, :, :] < v[:, None, :]).any(-1)
    result[valid] = ~dominates.any(1)
    return result


def aggregate(raw, expected):
    if raw.empty:
        return pd.DataFrame()
    grouped = raw.groupby(KEYS, dropna=False)
    summary = grouped[METRICS].mean().reset_index()
    summary["n"] = grouped.size().to_numpy()
    for metric in ("shared_MIG", "shared_modularity", "shared_tree_r2"):
        summary[metric+"_n"] = grouped[metric].count().to_numpy()
    summary["complete"] = summary["n"] == expected
    summary["cell_id"] = ["-".join(str(row[k]) for k in KEYS) for _, row in summary.iterrows()]
    return summary


def front(summary, task, scope, metric, settings, budget="actual_steps"):
    table = summary[summary.task == task].copy()
    if scope != "pooled":
        table = table[table.family == scope].copy()
    if table.empty:
        return table
    feasible = table.complete.copy()
    feasible &= table.shared_active_fraction >= settings["active_min"]
    if task == "balance":
        feasible &= table.survival >= settings["survival_min"]
        performance, direction = "J_safe", 1
    else:
        feasible &= (table.success >= settings["swingup_min"]) & (table.balanced_final >= settings["balanced_min"])
        performance, direction = "success", -1
    if metric == "tree":
        feasible &= table.shared_tree_r2 >= settings["tree_fidelity_min"]
        feasible &= table.shared_tree_r2_n == table.n
        interpretation, interpretation_direction = "shared_tree_leaves", 1
    else:
        interpretation, interpretation_direction = "shared_"+metric, -1
        feasible &= table[interpretation+"_n"] == table.n
    values = table[[performance, "actor_effective", interpretation, budget]].to_numpy(float)
    values *= np.array([direction, 1, interpretation_direction, 1])
    valid = feasible.to_numpy() & np.isfinite(values).all(1)
    table["feasible"] = valid
    membership = np.zeros(len(table), dtype=bool)
    membership[valid] = nondominated(values[valid])
    table["pareto"] = membership
    table["scope"], table["interpretability_objective"] = scope, metric
    table["performance_objective"] = performance
    table["budget_objective"] = budget
    return table


def collect(root, c):
    root = Path(root)
    manifest = make_manifest(c)
    reference_dir = evaluate_references(c, root)
    ref_cost = {}
    for task in c["tasks"]:
        records = read_json(reference_dir/(task+".json"))["episodes"]
        ref_cost[task] = np.mean([r["J_safe"] for r in records if r["condition"] == "disturbed"])
    rows, statuses = [], []
    for spec in manifest["runs"]:
        run_dir = root/"runs"/spec["run_id"]
        status = read_json(run_dir/"status.json") if (run_dir/"status.json").exists() else {"status": "missing"}
        statuses.append({**spec, **status})
        if not (run_dir/"checkpoints.json").exists():
            continue
        metadata = read_json(run_dir/"metadata.json")
        checkpoint_map = read_json(run_dir/"checkpoints.json")
        for requested in c["budgets"]:
            filename = checkpoint_map.get(str(requested))
            if filename is None:
                continue
            stem = Path(filename).stem
            ev_dir, ip_dir = run_dir/"evaluation"/stem, run_dir/"interpretability"/stem
            if not (ev_dir/"complete.json").exists() or not (ip_dir/"complete.json").exists():
                continue
            records = read_json(ev_dir/"episodes.json")
            episodes = [row for row in records if row["condition"] == "disturbed"]
            clean_records = [row for row in records if row["condition"] == "clean"]
            checkpoint_meta = read_json((run_dir/filename).with_suffix(".json"))
            row = {**spec, "sparsity": -1. if spec["sparsity"] is None else spec["sparsity"], "requested_budget": requested,
                   "actual_steps": checkpoint_meta["steps"], "training_seconds": checkpoint_meta["training_seconds"]}
            for key in ("J_safe", "success", "balanced_final", "survival", "recovery_seconds", "return", "force_rms", "saturation", "chattering", "capture_steps"):
                vals = [r[key] for r in episodes if r[key] is not None]
                row[key] = np.mean(vals) if vals else np.nan
            row["cost_ratio"] = row["J_safe"]/ref_cost[spec["task"]] if ref_cost[spec["task"]] > 1e-12 else np.nan
            for key in ("J_safe", "success", "survival"):
                row["clean_"+key] = np.mean([r[key] for r in clean_records])
            row["disturbance_cost_increase"] = row["J_safe"]-row["clean_J_safe"]
            for key in ("actor_effective", "actor_stored", "core_unique_edges"):
                row[key] = metadata["architecture"][key]
            for mode in ("shared", "on_policy"):
                scores = read_json(ip_dir/mode/"metrics.json")
                for key in ("MIG", "modularity", "tree_r2", "tree_leaves", "active_fraction"):
                    row[mode+"_"+key] = scores[key]
            rows.append(row)
    return pd.DataFrame(rows), statuses


def matched_comparisons(summary, settings):
    rows = []
    for _, group in summary[summary.complete].groupby(["task", "size", "requested_budget"]):
        for (_, a), (_, b) in itertools.combinations(group.iterrows(), 2):
            if a["task"] == "balance":
                distance = abs(a.J_safe-b.J_safe)/max(abs(a.J_safe), abs(b.J_safe), 1e-12)
                supported = distance <= settings["match_relative_cost"]
            else:
                distance = abs(a.success-b.success)
                supported = distance <= settings["match_success"]
            rows.append({"task": a["task"], "size": a["size"], "budget": a.requested_budget,
                         "a": a.cell_id, "b": b.cell_id, "performance_distance": distance, "matched": supported,
                         "MIG_difference": a.shared_MIG-b.shared_MIG, "modularity_difference": a.shared_modularity-b.shared_modularity,
                         "tree_r2_difference": a.shared_tree_r2-b.shared_tree_r2})
    return pd.DataFrame(rows, columns=["task", "size", "budget", "a", "b", "performance_distance", "matched", "MIG_difference", "modularity_difference", "tree_r2_difference"])


def plots(summary, c, directory):
    for task in c["tasks"]:
        subset = summary[summary.task == task]
        performance = "J_safe" if task == "balance" else "success"
        for metric in (performance, "shared_MIG", "shared_modularity"):
            fig, axes = plt.subplots(len(c["families"]), len(c["wirings"]), squeeze=False, figsize=(8, max(3, 1.9*len(c["families"]))))
            for i, family in enumerate(c["families"]):
                for j, wiring in enumerate(c["wirings"]):
                    table = subset[(subset.family == family) & (subset.wiring == wiring)]
                    grid = table.pivot_table(index="size", columns="requested_budget", values=metric, aggfunc="mean").reindex(index=c["sizes"], columns=c["budgets"])
                    im = axes[i, j].imshow(grid.to_numpy(float), aspect="auto", origin="lower")
                    axes[i, j].set_title(f"{family} / {wiring}")
                    axes[i, j].set_xticks(range(len(c["budgets"])), [f"{b/1000:g}k" for b in c["budgets"]], fontsize=7)
                    axes[i, j].set_yticks(range(len(c["sizes"])), c["sizes"], fontsize=7)
                    fig.colorbar(im, ax=axes[i, j], fraction=.04)
            fig.suptitle(f"{c['label']}: {task} / {metric}; seed means")
            fig.tight_layout()
            fig.savefig(directory/f"{task}_{metric}_heatmap.png", dpi=140)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for family, group in subset.groupby("family"):
            axes[0].scatter(group[performance], group.shared_MIG, label=family)
            axes[1].scatter(group.shared_tree_r2, group.on_policy_tree_r2, label=family)
        axes[0].set(xlabel=performance, ylabel="Shared-probe MIG")
        axes[1].set(xlabel="Shared tree test R²", ylabel="On-policy tree test R²")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{c['label']}: {task} — descriptive, not causal")
        fig.tight_layout()
        fig.savefig(directory/f"{task}_interpretability.png", dpi=140)
        plt.close(fig)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(projection="3d")
        cloud = ax.scatter(subset.actor_effective, subset[performance], subset.shared_MIG, c=subset.actual_steps, cmap="viridis")
        ax.set(xlabel="Effective actor parameters", ylabel=performance, zlabel="Shared MIG")
        fig.colorbar(cloud, ax=ax, label="Training transitions", shrink=.6)
        ax.set_title(f"{c['label']}: {task}; projection of four objectives")
        fig.savefig(directory/f"{task}_tradeoffs.png", dpi=140)
        plt.close(fig)


def analyze(root, c):
    directory = Path(root)/"analysis"
    directory.mkdir(parents=True, exist_ok=True)
    raw, statuses = collect(root, c)
    write_json(directory/"run_statuses.json", statuses)
    if raw.empty:
        raise ValueError("No evaluated and interpreted checkpoints. Run evaluate and interpret first.")
    for key in METRICS:
        raw[key] = pd.to_numeric(raw[key], errors="coerce")
    raw.to_csv(directory/"per_seed.csv", index=False)
    summary = aggregate(raw, len(c["seeds"]))
    scopes = ["pooled", *c["families"]]
    settings = c["analysis"]
    def fronts(table):
        parts = [front(table, task, scope, metric, settings, budget) for task in c["tasks"] for scope in scopes
                 for metric in ("MIG", "modularity", "tree") for budget in settings.get("budget_axes", ["actual_steps"])]
        return pd.concat(parts, ignore_index=True)
    point_fronts = fronts(summary)
    rng = np.random.default_rng(seed_for(c["seed"], "analysis_bootstrap"))
    sampled_metrics, memberships, feasible_counts = [], {}, {}
    original_complete = set(summary.loc[summary.complete, "cell_id"])
    for _ in range(settings["bootstrap"]):
        drawn = rng.choice(c["seeds"], size=len(c["seeds"]), replace=True)
        sampled = pd.concat([raw[raw.seed == s] for s in drawn], ignore_index=True)
        boot = aggregate(sampled, len(c["seeds"]))
        boot["complete"] &= boot.cell_id.isin(original_complete)
        sampled_metrics.append(boot.set_index("cell_id")[METRICS])
        for row in boot.itertuples():
            meets = row.survival >= settings["survival_min"]
            meets &= row.cost_ratio <= settings.get("balance_cost_ratio_max", 2.) if row.task == "balance" else row.success >= settings["swingup_min"] and row.balanced_final >= settings["balanced_min"]
            feasible_counts[row.cell_id] = feasible_counts.get(row.cell_id, 0)+int(meets and row.complete)
        for row in fronts(boot).itertuples():
            key = (row.cell_id, row.scope, row.interpretability_objective, row.budget_objective)
            memberships[key] = memberships.get(key, 0) + int(row.pareto)
    for metric in METRICS:
        estimates = pd.DataFrame({i: table[metric] for i, table in enumerate(sampled_metrics)})
        low, high = estimates.quantile(.025, axis=1), estimates.quantile(.975, axis=1)
        summary[metric+"_ci_low"] = summary.cell_id.map(low)
        summary[metric+"_ci_high"] = summary.cell_id.map(high)
    summary["performance_feasibility_frequency"] = summary.cell_id.map(lambda key: feasible_counts.get(key, 0)/settings["bootstrap"])
    point_fronts["membership_frequency"] = [memberships.get((r.cell_id, r.scope, r.interpretability_objective, r.budget_objective), 0)/settings["bootstrap"] for r in point_fronts.itertuples()]
    summary.to_csv(directory/"summary.csv", index=False)
    point_fronts.to_csv(directory/"frontier_candidates.csv", index=False)
    point_fronts[point_fronts.pareto].to_csv(directory/"pareto.csv", index=False)
    matches = matched_comparisons(summary, settings)
    matches.to_csv(directory/"performance_matches.csv", index=False)
    minima = []
    for key, group in summary.groupby(["task", "family", "wiring", "requested_budget"]):
        valid = group.complete & (group.survival >= settings["survival_min"])
        if key[0] == "swingup":
            valid &= (group.success >= settings["swingup_min"]) & (group.balanced_final >= settings["balanced_min"])
        else:
            valid &= group.cost_ratio <= settings.get("balance_cost_ratio_max", 2.)
        candidate = group[valid].sort_values(["size", "actor_effective"])
        confident = candidate[candidate.performance_feasibility_frequency >= .95]
        minima.append(dict(zip(["task", "family", "wiring", "requested_budget"], key), minimum_neurons=None if candidate.empty else int(candidate.iloc[0]["size"]),
                           minimum_neurons_probability95=None if confident.empty else int(confident.iloc[0]["size"])))
    pd.DataFrame(minima).to_csv(directory/"minimum_sizes.csv", index=False)
    interaction_rows = []
    for key, group in summary.groupby(["task", "family", "wiring", "sparsity"]):
        metric, sign = ("J_safe", -1) if key[0] == "balance" else ("success", 1)
        sizes, budgets = sorted(group["size"].unique()), sorted(group.requested_budget.unique())
        lookup = {(r["size"], r.requested_budget): r for _, r in group.iterrows()}
        for small, large in zip(sizes, sizes[1:]):
            for early, late in zip(budgets, budgets[1:]):
                cells = [lookup.get(pair) for pair in [(small, late), (large, late), (small, early), (large, early)]]
                if any(cell is None or not cell.complete for cell in cells):
                    continue
                weights = np.array([1, -1, -1, 1])*sign
                effect = sum(w*cell[metric] for w, cell in zip(weights, cells))
                estimates = [sum(w*sample.loc[cell.cell_id, metric] for w, cell in zip(weights, cells)) for sample in sampled_metrics]
                interaction_rows.append(dict(zip(["task", "family", "wiring", "sparsity"], key), small=small, large=large, early=early, late=late,
                                             small_relative_improvement=effect, ci_low=np.quantile(estimates, .025), ci_high=np.quantile(estimates, .975)))
    pd.DataFrame(interaction_rows, columns=["task", "family", "wiring", "sparsity", "small", "large", "early", "late", "small_relative_improvement", "ci_low", "ci_high"]).to_csv(directory/"budget_size_interactions.csv", index=False)
    plots(summary, c, directory)
    write_json(directory/"analysis_settings.json", {"settings": settings, "label": c["label"], "bootstrap_unit": "whole training seed, with all budgets retained", "matched_pairs": int(matches.matched.sum()),
                                                  "scope": "Canonical physics factors; associations are descriptive. Dense/structured depth differs. No significance tests performed."})
    (directory/"REPORT.md").write_text(f"# Cart pole results\n\nLabel: **{c['label']}**. " + ("These are software smoke checks, not scientific evidence. " if c["label"] == "smoke" else "") +
        f"Analyzed {len(raw)} seed checkpoints in {len(summary)} cells; {int(summary.complete.sum())} have all configured seeds.\n\n"
        "See summary.csv for seed means and 95% whole-seed bootstrap intervals, pareto.csv for feasible non-dominated sets, and performance_matches.csv for matched comparisons and their support. "
        "Bootstrap intervals measure training-seed uncertainty conditional on this fixed evaluation suite. They do not include new test-suite uncertainty. "
        "Minimum sizes use point-estimate performance thresholds, not a proof of capacity. Failed/missing run statuses are recorded separately.\n")
    return {"seed_checkpoints": len(raw), "cells": len(summary), "frontier_rows": int(point_fronts.pareto.sum()), "matched_pairs": int(matches.matched.sum())}
