"""Paired physical evaluation, replay datasets, and reproducible shared probes."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import digest, seed_for
from .envs import ANGLE_LIMIT, Q, R, CartPole, Reference
from .io import read_json, write_json
from .training import load_agent


def first_dwell(band, length):
    count = 0
    for i, valid in enumerate(band):
        count = count+1 if valid else 0
        if count >= length:
            return i-length+1
    return None


def failure_aware_cost(stage_costs, horizon, failure_cost, failed):
    costs = np.full(horizon, failure_cost, dtype=float)
    costs[:len(stage_costs)] = np.minimum(stage_costs, failure_cost)
    if failed and len(stage_costs):
        costs[len(stage_costs)-1:] = failure_cost
    return float(costs.mean())


def episode(task, c, seed, agent=None, controller=None, disturbed=True, excitation=None):
    env_args = dict(c["env"])
    if not disturbed:
        env_args["disturbance"] = 0.
    env = CartPole(task, **env_args)
    obs, initial = env.reset(seed=seed)
    state = agent.actor.initial_state(1) if agent is not None else None
    fields = {k: [] for k in ("observations", "states", "next_states", "actions", "forces", "disturbances", "rewards")}
    responses = []
    terminated = truncated = False
    smooth_noise = 0.
    while not (terminated or truncated):
        before = env.state.copy()
        if agent is not None:
            with torch.no_grad():
                mean, state, activation = agent.actor.step(torch.from_numpy(obs)[None], state)
                action = mean.tanh().numpy()[0]
                responses.append(activation.numpy()[0])
        else:
            action = controller(env.state)
        if excitation is not None:
            smooth_noise = .9*smooth_noise + excitation.normal(0, .06)
            action = np.clip(action + smooth_noise, -1, 1)
        next_obs, reward, terminated, truncated, info = env.step(action)
        for key, value in dict(observations=obs, states=before, next_states=info["state"], actions=action,
                               forces=info["force"], disturbances=info["disturbance"], rewards=reward).items():
            fields[key].append(value)
        obs = next_obs
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    if responses:
        arrays["activations"] = np.asarray(responses)
    physical = arrays["next_states"]
    horizon = c["env"]["horizon"]
    ev = c["evaluation"]
    band = (np.abs(physical[:, 2]) < ANGLE_LIMIT) & (np.abs(physical[:, 0]) < 2.4)
    capture = first_dwell(band, ev["dwell"])
    stages = np.einsum("ni,ij,nj->n", physical, Q, physical) + R*arrays["forces"]**2
    recovery, censored = [], []
    for index, (pulse, amplitude) in enumerate(initial["pulses"]):
        if not amplitude:
            continue
        end = pulse + c["env"]["pulse_duration"]
        stop = initial["pulses"][index+1][0] if index+1 < len(initial["pulses"]) else horizon
        recovery_band = (np.abs(physical[:, 0]) < .5) & (np.abs(physical[:, 2]) < .05) & (np.abs(physical[:, 1]) < .5) & (np.abs(physical[:, 3]) < .5)
        offset = first_dwell(recovery_band[end:min(stop, len(physical))], ev["recovery_dwell"])
        recovery.append((stop-end if offset is None else offset)*env.dt)
        censored.append(offset is None)
    force = arrays["forces"]
    after_capture = stages[capture:] if capture is not None else np.array([])
    metrics = {"seed": seed, "condition": "disturbed" if disturbed else "clean", "length": len(physical),
               "survival": int(not terminated), "failed": int(terminated), "return": float(arrays["rewards"].sum()),
               "cost": float(stages.mean()), "J_safe": failure_aware_cost(stages, horizon, ev["failure_cost"], terminated),
               "cost_clip_fraction": float(np.mean(stages > ev["failure_cost"])),
               "success": int(capture is not None), "capture_steps": horizon if capture is None else capture,
               "capture_censored": capture is None, "balanced_final": int(not terminated and len(band) >= ev["final_window"] and band[-ev["final_window"]:].all()),
               "upright_occupancy": float(band.sum()/horizon), "balance_cost_conditional": float(after_capture.mean()) if after_capture.size else None,
               "force_rms": float(np.sqrt(np.mean(force**2))), "force_peak": float(np.max(np.abs(force))),
               "saturation": float(np.mean(np.abs(arrays["actions"]) >= .99)), "chattering": float(np.abs(np.diff(force)).mean()) if len(force)>1 else 0.,
               "work": float(np.sum(np.abs(force*physical[:, 1]))*env.dt),
               "recovery_seconds": float(np.mean(recovery)) if recovery else None,
               "recovery_censored_fraction": float(np.mean(censored)) if censored else None,
               "pulses": initial["pulses"], "initial_state": arrays["states"][0].tolist()}
    return arrays, metrics


def concatenate(trajectories):
    keys = set.intersection(*(set(x) for x in trajectories))
    result = {key: np.concatenate([trajectory[key] for trajectory in trajectories]) for key in keys}
    result["episode"] = np.concatenate([np.full(len(t["states"]), i, dtype=np.int32) for i, t in enumerate(trajectories)])
    return result


def evaluate_checkpoint(path, force=False):
    path = Path(path)
    directory = path.parent/"evaluation"/path.stem
    if (directory/"complete.json").exists() and not force:
        return read_json(directory/"complete.json")
    agent, ckpt = load_agent(path)
    c, spec = ckpt["config"], ckpt["spec"]
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for disturbed in (False, True):
        trajectories = []
        for i in range(c["evaluation"]["episodes"]):
            arrays, metrics = episode(spec["task"], c, seed_for(c["seed"], "test", spec["task"], i), agent=agent, disturbed=disturbed)
            trajectories.append(arrays)
            metrics["episode"] = i
            all_metrics.append(metrics)
        np.savez_compressed(directory/("disturbed.npz" if disturbed else "clean.npz"), **concatenate(trajectories))
    write_json(directory/"episodes.json", all_metrics)
    result = {"status": "complete", "steps": ckpt["steps"], "episodes": len(all_metrics), "seconds": time.perf_counter()-start}
    write_json(directory/"complete.json", result)
    return result


def evaluate_references(c, root, force=False):
    identity = digest({k: c[k] for k in ("seed", "env", "evaluation")})[:12]
    root = Path(root)/"references"/identity
    for task in c["tasks"]:
        output = root/(task+".json")
        if output.exists() and not force:
            continue
        ref = Reference(task, c["env"]["dt"], c["env"]["force_limit"])
        episodes = []
        for disturbed in (False, True):
            for i in range(c["evaluation"]["episodes"]):
                _, metrics = episode(task, c, seed_for(c["seed"], "test", task, i), controller=ref, disturbed=disturbed)
                metrics["episode"] = i
                episodes.append(metrics)
        write_json(output, {"gain": ref.gain, "episodes": episodes})
    return root


def make_probes(c, root, force=False):
    identity = digest({k: c[k] for k in ("seed", "env", "probe")})[:12]
    directory = Path(root)/"probes"/identity
    directory.mkdir(parents=True, exist_ok=True)
    for task in c["tasks"]:
        path = directory/(task+".npz")
        if path.exists() and not force:
            continue
        ref = Reference(task, c["env"]["dt"], c["env"]["force_limit"])
        trajectories = []
        for i in range(c["probe"]["episodes"]):
            excitation = np.random.default_rng(seed_for(c["seed"], "excitation", task, i)) if i % 2 else None
            data, _ = episode(task, c, seed_for(c["seed"], "probe", task, i), controller=ref, excitation=excitation)
            trajectories.append(data)
        bank = concatenate(trajectories)
        episodes = c["probe"]["episodes"]
        order = np.random.default_rng(seed_for(c["seed"], "probe_split", task)).permutation(episodes)
        split = np.full(episodes, 2, dtype=np.int32)
        split[order[:episodes//2]] = 0
        split[order[episodes//2:episodes//2+max(1, episodes//5)]] = 1
        bank["split"] = split[bank["episode"]]
        np.savez_compressed(path, **bank)
        write_json(path.with_suffix(".json"), {"probe_id": identity, "task": task, "episode_splits": split,
                                              "split_names": ["train", "validation", "test"], "config": c["probe"],
                                              "sources": "Alternating physical reference and reference plus smooth seeded excitation"})
    return directory
