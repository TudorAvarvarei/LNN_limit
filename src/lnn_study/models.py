"""Official dense cells and explicitly labeled masked NCP-template extensions."""
from __future__ import annotations

import math

import numpy as np
import torch
from ncps.torch import CfC, LTC
from ncps.torch.cfc_cell import CfCCell
from ncps.torch.ltc_cell import LTCCell
from ncps.wirings import AutoNCP
from torch import nn
from torch.nn import functional as F


def map_state(function, state):
    return tuple(map_state(function, s) for s in state) if isinstance(state, tuple) else function(state)


def reset_state(state, starts):
    return map_state(lambda s: s * (~starts).to(s.dtype).unsqueeze(-1), state)


class MaskedCell(nn.Module):
    """PyTorch RNN/GRU/LSTM equations; masks apply on every forward pass."""
    def __init__(self, family, inputs, hidden, input_mask, recurrent_mask):
        super().__init__()
        self.family = family
        self.cell = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}[family](inputs, hidden)
        gates = {"rnn": 1, "gru": 3, "lstm": 4}[family]
        self.register_buffer("mask_ih", torch.as_tensor(input_mask, dtype=torch.float32).repeat(gates, 1))
        self.register_buffer("mask_hh", torch.as_tensor(recurrent_mask, dtype=torch.float32).repeat(gates, 1))

    def forward(self, x, state):
        h = state[0] if self.family == "lstm" else state
        a = F.linear(x, self.cell.weight_ih * self.mask_ih, self.cell.bias_ih)
        b = F.linear(h, self.cell.weight_hh * self.mask_hh, self.cell.bias_hh)
        if self.family == "rnn":
            return torch.tanh(a+b)
        if self.family == "gru":
            ar, az, an = a.chunk(3, -1)
            br, bz, bn = b.chunk(3, -1)
            r, z = torch.sigmoid(ar+br), torch.sigmoid(az+bz)
            n = torch.tanh(an + r*bn)  # PyTorch resets AFTER the recurrent affine.
            return (1-z)*n + z*h
        i, f, g, o = (a+b).chunk(4, -1)
        c = torch.sigmoid(f)*state[1] + torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c), c


class MaskedLinear(nn.Linear):
    def __init__(self, mask):
        super().__init__(mask.shape[1], mask.shape[0])
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, x):
        return F.linear(x, self.weight*self.mask, self.bias)


class Actor(nn.Module):
    def __init__(self, spec, model_config, dt=.02):
        super().__init__()
        self.spec, self.dt = spec, dt
        self.family, self.size = spec["family"], spec["size"]
        self.inputs = 4 if spec["task"] == "balance" else 5
        self.register_buffer("observation_scale", torch.tensor([2.4, 3., .21, 3.] if self.inputs == 4 else [2.4, 3., 1., 1., 3.]))
        self.wiring = None
        self.groups = [list(range(self.size))]
        if spec["wiring"] == "ncp":
            self.wiring = AutoNCP(self.size, model_config["motor_neurons"], sparsity_level=spec["sparsity"], seed=spec["topology_seed"])
            self.wiring.build(self.inputs)
            self.groups = [list(self.wiring.get_neurons_of_layer(i)) for i in range(self.wiring.num_layers)]
        if self.family in ("cfc", "ltc"):
            units = self.wiring if self.wiring is not None else self.size
            args = {"backbone_units": 0, "backbone_layers": 0} if self.family == "cfc" and self.wiring is None else {}
            if self.family == "ltc":
                args.update(ode_unfolds=model_config["ode_unfolds"], implicit_param_constraints=True)
            self.core = {"cfc": CfC, "ltc": LTC}[self.family](self.inputs, units, mixed_memory=False, **args)
            outputs = self.core.output_size
        elif self.wiring is None:
            self.core = nn.Linear(self.inputs, self.size) if self.family == "mlp" else {"gru": nn.GRUCell, "lstm": nn.LSTMCell, "rnn": nn.RNNCell}[self.family](self.inputs, self.size)
            outputs = self.size
        else:
            layers = []
            previous = None
            for group in self.groups:
                incoming = self.wiring.sensory_adjacency_matrix[:, group] if previous is None else self.wiring.adjacency_matrix[np.ix_(previous, group)]
                input_mask = np.abs(incoming.T)
                if self.family == "mlp":
                    layers.append(MaskedLinear(input_mask))
                else:
                    recurrent = np.abs(self.wiring.adjacency_matrix[np.ix_(group, group)].T)
                    np.fill_diagonal(recurrent, 1)  # explicit learned self edges, beyond intrinsic gate memory
                    layers.append(MaskedCell(self.family, input_mask.shape[1], len(group), input_mask, recurrent))
                previous = group
            self.core = nn.ModuleList(layers)
            outputs = len(self.groups[-1])
        self.head = nn.Linear(outputs, 1)
        nn.init.orthogonal_(self.head.weight, gain=.01)
        nn.init.zeros_(self.head.bias)
        self.log_std = nn.Parameter(torch.full((1,), -.5))

    def initial_state(self, batch):
        width = 0 if self.family == "mlp" else self.size
        h = torch.zeros(batch, width, device=self.observation_scale.device)
        return (h, h.clone()) if self.family == "lstm" else h

    def step(self, observations, state, starts=None):
        if starts is not None:
            state = reset_state(state, starts)
        x = (observations/self.observation_scale).clamp(-10, 10)
        if self.family in ("cfc", "ltc"):
            out, state = self.core.rnn_cell(x, state, self.dt)
            activations = state
        elif self.wiring is None:
            if self.family == "mlp":
                out = activations = torch.tanh(self.core(x))
            else:
                state = self.core(x, state)
                out = activations = state[0] if isinstance(state, tuple) else state
        else:
            hs, cs, activations = [], [], []
            offset = 0
            for layer, group in zip(self.core, self.groups):
                width = len(group)
                if self.family == "mlp":
                    x = torch.tanh(layer(x))
                elif self.family == "lstm":
                    x, cell = layer(x, (state[0][:, offset:offset+width], state[1][:, offset:offset+width]))
                    hs.append(x)
                    cs.append(cell)
                else:
                    x = layer(x, state[:, offset:offset+width])
                    hs.append(x)
                activations.append(x)
                offset += width
            out, activations = x, torch.cat(activations, -1)
            if hs:
                state = (torch.cat(hs, -1), torch.cat(cs, -1)) if cs else torch.cat(hs, -1)
        return self.head(out), state, activations

    def sequence(self, observations, state, starts):
        means, responses = [], []
        for x, reset in zip(observations, starts):
            mean, state, activation = self.step(x, state, reset)
            means.append(mean)
            responses.append(activation)
        return torch.stack(means), state, torch.stack(responses)

    def distribution(self, means):
        return torch.distributions.Normal(means, self.log_std.clamp(-5, 2).exp())


def squashed_log_prob(distribution, latent):
    log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2*latent))
    return (distribution.log_prob(latent) - log_jacobian).sum(-1)


class Agent(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        self.actor = Actor(spec, config["model"], config["env"]["dt"])
        width = config["model"]["critic_width"]
        self.critic = nn.Sequential(nn.Linear(self.actor.inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def value(self, observations):
        return self.critic((observations/self.actor.observation_scale).clamp(-10, 10)).squeeze(-1)


def architecture(actor, critic=None):
    """Audit each native/custom pathway, including unmasked CfC time gates."""
    masks = {}
    for prefix, module in actor.named_modules():
        if isinstance(module, MaskedCell):
            masks[prefix+".cell.weight_ih"] = module.mask_ih
            masks[prefix+".cell.weight_hh"] = module.mask_hh
        elif isinstance(module, MaskedLinear):
            masks[prefix+".weight"] = module.mask
        elif isinstance(module, CfCCell) and module.sparsity_mask is not None:
            masks[prefix+".ff1.weight"] = module.sparsity_mask
            masks[prefix+".ff2.weight"] = module.sparsity_mask
        elif isinstance(module, LTCCell):
            for name in ("w", "sigma", "mu", "erev"):
                masks[prefix+"."+name] = module._params["sparsity_mask"]
                masks[prefix+".sensory_"+name] = module._params["sensory_sparsity_mask"]
    audit = []
    for name, parameter in actor.named_parameters():
        effective = int(torch.count_nonzero(masks[name])) if name in masks else parameter.numel()
        audit.append({"name": name, "shape": list(parameter.shape), "stored": parameter.numel(),
                      "trainable": parameter.requires_grad, "effective": effective if parameter.requires_grad else 0,
                      "mask": masks[name].detach().cpu().tolist() if name in masks else None})
    core = [p for p in audit if p["name"].startswith("core.")]
    # Unique directed neuron pairs are distinct from per-gate parameter multiplicity.
    if actor.family == "cfc":
        if actor.wiring is None:
            edges = actor.size*(actor.inputs+actor.size)
        else:
            previous, edges = actor.inputs, 0
            for group in actor.groups:
                edges += len(group)*(previous+len(group))  # dense time gates
                previous = len(group)
    elif actor.family == "ltc":
        cell = actor.core.rnn_cell
        edges = int(torch.count_nonzero(cell._params["sparsity_mask"]) + torch.count_nonzero(cell._params["sensory_sparsity_mask"]))
    elif actor.wiring is None:
        edges = actor.inputs*actor.size + (0 if actor.family == "mlp" else actor.size**2)
    else:
        edges = 0
        for layer in actor.core:
            edges += int(torch.count_nonzero(layer.mask)) if isinstance(layer, MaskedLinear) else int(torch.count_nonzero(layer.mask_ih[:layer.cell.hidden_size]) + torch.count_nonzero(layer.mask_hh[:layer.cell.hidden_size]))
    order = [i for group in actor.groups for i in group]
    if actor.family == "ltc":
        order = list(range(actor.size))
    return {"implementation": "native_ncps" if actor.family in ("cfc", "ltc") else ("native_pytorch" if actor.wiring is None else ("ncp_feedforward_projection" if actor.family == "mlp" else "ncp_template_extension")),
            "neurons": actor.size, "state_scalars": 0 if actor.family == "mlp" else actor.size*(2 if actor.family == "lstm" else 1),
            "core_effective": sum(p["effective"] for p in core), "actor_effective": sum(p["effective"] for p in audit),
            "actor_stored": sum(p["stored"] for p in audit), "actor_trainable": sum(p["stored"] for p in audit if p["trainable"]),
            "actor_buffer_scalars": sum(b.numel() for b in actor.buffers()),
            "actor_state_bytes": sum(t.numel()*t.element_size() for t in actor.state_dict().values()),
            "critic_parameters": sum(p.numel() for p in critic.parameters()) if critic is not None else None,
            "core_unique_edges": edges, "head_edges": actor.head.weight.numel(), "activation_neuron_ids": order,
            "groups": actor.groups, "pathways": audit,
            "wiring": None if actor.wiring is None else {"sensory": actor.wiring.sensory_adjacency_matrix.tolist(), "recurrent": actor.wiring.adjacency_matrix.tolist()},
            "notes": "Unique edges exclude intrinsic carry memory. CfC time gates remain dense. GRU/LSTM extensions use topology without polarity constraints."}
