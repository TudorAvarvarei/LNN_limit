"""Held-out neuron trees and canonical common-physics MIG/modularity."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from .evaluation import make_probes
from .io import write_json
from .training import load_agent


def edges_fit(values, bins):
    return np.unique(np.quantile(values, np.linspace(0, 1, bins+1)))[1:-1]


def entropy(values):
    _, counts = np.unique(values, return_counts=True)
    p = counts/counts.sum()
    return float(-np.sum(p*np.log(p)))


def mutual_information(x, y):
    _, x = np.unique(x, return_inverse=True)
    _, y = np.unique(y, return_inverse=True)
    joint = np.bincount(x*(y.max()+1)+y, minlength=(x.max()+1)*(y.max()+1)).reshape(x.max()+1, y.max()+1).astype(float)
    joint /= joint.sum()
    product = joint.sum(1, keepdims=True)*joint.sum(0, keepdims=True)
    valid = joint > 0
    return float(np.sum(joint[valid]*np.log(joint[valid]/product[valid])))


def conditional_mi(x, y, z):
    return sum(np.mean(z == v)*mutual_information(x[z == v], y[z == v]) for v in np.unique(z))


def disentanglement(train_z, test_z, train_factors, test_factors, bins=10):
    z_edges = [edges_fit(column, bins) for column in train_z.T]
    f_edges = [edges_fit(column, bins) for column in train_factors.T]
    z = np.array([np.digitize(column, edge) for column, edge in zip(test_z.T, z_edges)]).T
    factors = np.array([np.digitize(column, edge) for column, edge in zip(test_factors.T, f_edges)]).T
    matrix = np.array([[mutual_information(a, b) for b in factors.T] for a in z.T])
    entropies = np.array([entropy(b) for b in factors.T])
    valid = entropies > 1e-12
    gaps = []
    if len(z_edges) >= 2:
        ordered = np.sort(matrix[:, valid], axis=0)
        gaps = (ordered[-1]-ordered[-2])/entropies[valid]
    mods = np.zeros(len(z_edges))
    if np.sum(valid) >= 2:
        M = matrix[:, valid]
        for i, row in enumerate(M):
            maximum = row.max()
            if maximum > 1e-12:
                mods[i] = 1 - (np.sum(row**2)-maximum**2)/((len(row)-1)*maximum**2)
    info = matrix.max(axis=1) > 1e-12
    return {"MIG": float(np.mean(gaps)) if len(gaps) else None,
            "modularity": float(mods.mean()) if np.sum(valid) >= 2 else None,
            "modularity_informative": float(mods[info].mean()) if info.any() and np.sum(valid)>=2 else None,
            "informative_fraction": float(info.mean()), "valid_factors": int(valid.sum()),
            "mi_matrix": matrix.tolist(), "factor_entropies": entropies.tolist(),
            "activation_edges": [e.tolist() for e in z_edges], "factor_edges": [e.tolist() for e in f_edges]}


def split_masks(episodes, split, burn_in):
    usable = np.ones(len(episodes), dtype=bool)
    for episode in np.unique(episodes):
        usable[np.flatnonzero(episodes == episode)[:burn_in]] = False
    return [(split == i) & usable for i in range(3)]


def score(bank, activations, cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train, validation, test = split_masks(bank["episode"], bank["split"], cfg["burn_in"])
    if train.sum() < 2 or test.sum() < 2:
        raise ValueError("Too few held-out probe samples; shorten burn-in or enlarge probe bank")
    factors = bank["states"].copy()  # canonical angle is ONE periodic factor
    neurons, trees = [], []
    for i, z in enumerate(activations.T):
        std, mean = np.std(z[train]), np.mean(z[train])
        if std <= cfg["activity_epsilon"]:
            neurons.append({"neuron": i, "active": False, "r2": None, "leaves": 1, "nodes": 1, "depth": 0})
            trees.append(None)
            continue
        tree = DecisionTreeRegressor(max_depth=cfg["tree_depth"], min_samples_leaf=cfg["min_leaf_fraction"], ccp_alpha=cfg["ccp_alpha"], random_state=0)
        tree.fit(factors[train], (z[train]-mean)/std)
        prediction = tree.predict(factors[test])
        truth = (z[test]-mean)/std
        r2 = float(r2_score(truth, prediction, force_finite=False)) if np.std(truth)>cfg["activity_epsilon"] else None
        val_truth = (z[validation]-mean)/std
        val_r2 = float(r2_score(val_truth, tree.predict(factors[validation]), force_finite=False)) if validation.sum()>1 and np.std(val_truth)>cfg["activity_epsilon"] else None
        neurons.append({"neuron": i, "active": True, "r2": r2, "validation_r2": val_r2,
                        "normalized_mse": float(np.mean((prediction-truth)**2)), "leaves": tree.get_n_leaves(),
                        "nodes": tree.tree_.node_count, "depth": tree.get_depth(), "target_mean": float(mean), "target_std": float(std),
                        "dominant_feature": ["x", "x_dot", "theta", "theta_dot"][int(np.argmax(tree.feature_importances_))]})
        trees.append(tree)
    defined = [n for n in neurons if n["active"] and n["r2"] is not None and np.isfinite(n["r2"])]
    coverage = sum(n["active"] for n in neurons)/len(neurons)
    result = disentanglement(activations[train], activations[test], factors[train], factors[test], cfg["bins"])
    result.update(tree_r2=float(np.mean([n["r2"] for n in defined])) if defined else None,
                  tree_leaves=float(np.mean([n["leaves"] for n in neurons])),
                  tree_nodes=float(np.mean([n["nodes"] for n in neurons])), active_fraction=coverage,
                  train_samples=int(train.sum()), validation_samples=int(validation.sum()), test_samples=int(test.sum()),
                  vocabulary=["x", "x_dot", "wrapped_theta", "theta_dot"], calibration="canonical_uncalibrated", neurons=neurons)
    result["tree_r2_coverage_weighted"] = result["tree_r2"]*coverage if result["tree_r2"] is not None else None
    write_json(directory/"metrics.json", result)
    joblib.dump(trees, directory/"trees.joblib")
    np.savez_compressed(directory/"activations.npz", activations=activations, episode=bank["episode"], split=bank["split"], train=train, test=test)
    return result


def interpret_checkpoint(path, root, force=False):
    path = Path(path)
    directory = path.parent/"interpretability"/path.stem
    if (directory/"complete.json").exists() and not force:
        return
    agent, ckpt = load_agent(path)
    c, spec = ckpt["config"], ckpt["spec"]
    probe_dir = make_probes(c, root)
    with np.load(probe_dir/(spec["task"]+".npz")) as data:
        bank = dict(data)
    activations = []
    with torch.no_grad():
        state = agent.actor.initial_state(1)
        previous = -1
        for obs, episode_id in zip(bank["observations"], bank["episode"]):
            if episode_id != previous:
                state = agent.actor.initial_state(1)
            _, state, z = agent.actor.step(torch.from_numpy(obs)[None], state)
            activations.append(z.numpy()[0])
            previous = episode_id
    score(bank, np.asarray(activations), c["probe"], directory/"shared")
    on_policy = path.parent/"evaluation"/path.stem/"disturbed.npz"
    if not on_policy.exists():
        raise FileNotFoundError("Run evaluation before interpreting on-policy responses")
    with np.load(on_policy) as data:
        policy_bank = dict(data)
    ids = np.unique(policy_bank["episode"])
    # At least three complete episode groups, never a random time-step split.
    splits = np.full(len(ids), 2, dtype=np.int32)
    splits[:max(1, len(ids)//2)] = 0
    splits[max(1, len(ids)//2):max(1, len(ids)//2)+max(1, len(ids)//5)] = 1
    policy_bank["split"] = splits[policy_bank["episode"]]
    score(policy_bank, policy_bank["activations"], c["probe"], directory/"on_policy")
    write_json(directory/"complete.json", {"status": "complete", "probe": str(probe_dir), "calibration": "Canonical MIG and modularity; no learned leaf-factor calibration"})
