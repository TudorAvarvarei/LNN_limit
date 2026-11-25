"""Continuous-force Gym cart-pole dynamics with explicit, replayable disturbances."""
from __future__ import annotations

import copy
import math

import gymnasium as gym
import numpy as np
from scipy.linalg import solve_discrete_are

Q = np.diag([2., 1., 20., 2.])
R = 0.05
ANGLE_LIMIT = math.radians(12)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def transition(state, force, dt=0.02):
    """Gym's uniform rod (half-length .5 m), semi-implicit Euler."""
    x, velocity, theta, omega = state
    sin, cos = np.sin(theta), np.cos(theta)
    temp = (force + 0.1 * 0.5 * omega**2 * sin) / 1.1
    alpha = (9.8 * sin - cos * temp) / (0.5 * (4 / 3 - 0.1 * cos**2 / 1.1))
    accel = temp - 0.1 * 0.5 * alpha * cos / 1.1
    velocity, omega = velocity + dt * accel, omega + dt * alpha
    return np.array([x + dt * velocity, velocity, wrap(theta + dt * omega), omega], dtype=np.float64)


def observation(state, task):
    x, v, theta, omega = state
    return np.asarray([x, v, wrap(theta), omega] if task == "balance" else [x, v, np.sin(theta), np.cos(theta), omega], dtype=np.float32)


class CartPole(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task="balance", horizon=500, dt=0.02, force_limit=10., disturbance=5., pulse_duration=3):
        if task not in ("balance", "swingup"):
            raise ValueError(task)
        self.task, self.horizon, self.dt = task, horizon, dt
        self.force_limit, self.disturbance, self.pulse_duration = force_limit, disturbance, pulse_duration
        self.action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4 if task == "balance" else 5,), dtype=np.float32)
        self.state, self.t, self.pulses = np.zeros(4), 0, []
        self.ended = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.state = self.np_random.uniform(-0.05, 0.05, 4)
        if self.task == "swingup":
            self.state[2] = wrap(np.pi + self.state[2])
        if "state" in options:
            self.state = np.asarray(options["state"], dtype=np.float64).copy()
        self.t, self.ended = 0, False
        # Disjoint windows guarantee separated pulses, even in short smoke episodes.
        windows = ((0.15, 0.35), (0.55, 0.75)) if self.task == "balance" else ((0.6, 0.72), (0.82, 0.9))
        self.pulses = [(int(self.np_random.integers(max(1, int(lo*self.horizon)), max(2, int(hi*self.horizon)))),
                        float(self.np_random.choice([-1, 1]) * self.disturbance)) for lo, hi in windows]
        if "pulses" in options:
            self.pulses = [tuple(p) for p in options["pulses"]]
        return observation(self.state, self.task), self.info(0., 0.)

    def info(self, force, disturbance):
        return {"state": self.state.copy(), "step": self.t, "force": force,
                "disturbance": disturbance, "pulses": list(self.pulses)}

    def step(self, action):
        if self.ended:
            raise RuntimeError("Reset after termination or truncation")
        action = float(np.asarray(action).reshape(-1)[0])
        if not np.isfinite(action):
            raise ValueError("Action must be finite")
        a = np.clip(action, -1., 1.)
        force = float(a * self.force_limit)
        external = sum(f for start, f in self.pulses if start <= self.t < start + self.pulse_duration)
        self.state = transition(self.state, force + external, self.dt)
        self.t += 1
        x, v, theta, omega = self.state
        terminated = bool(abs(x) > 2.4 or (self.task == "balance" and abs(wrap(theta)) > ANGLE_LIMIT))
        truncated = bool(self.t >= self.horizon and not terminated)
        reward = max(0., 1 - .05 * (self.state @ Q @ self.state) - .05 * a*a) if self.task == "balance" else max(0., .5*(1+np.cos(theta)) - .1*x*x - .01*omega*omega - .05*a*a)
        self.ended = terminated or truncated
        return observation(self.state, self.task), float(reward), terminated, truncated, self.info(force, external)

    def snapshot(self):
        return {"state": self.state.copy(), "t": self.t, "pulses": self.pulses,
                "ended": self.ended, "rng": copy.deepcopy(self.np_random.bit_generator.state)}

    def restore(self, data):
        self.state, self.t = data["state"].copy(), data["t"]
        self.pulses, self.ended = copy.deepcopy(data["pulses"]), data["ended"]
        self.np_random.bit_generator.state = data["rng"]


class Reference:
    """Numerically linearized discrete LQR plus energy pumping to a capture region."""
    def __init__(self, task, dt=.02, force_limit=10.):
        self.task, self.dt, self.force_limit = task, dt, force_limit
        eps = 1e-5
        A = np.column_stack([(transition(np.eye(4)[i]*eps, 0, dt) - transition(-np.eye(4)[i]*eps, 0, dt))/(2*eps) for i in range(4)])
        B = ((transition(np.zeros(4), eps, dt) - transition(np.zeros(4), -eps, dt))/(2*eps))[:, None]
        P = solve_discrete_are(A, B, Q, np.array([[R]]))
        self.gain = np.linalg.solve(np.array([[R]]) + B.T @ P @ B, B.T @ P @ A)[0]

    def __call__(self, state):
        x, v, theta, omega = state
        theta = wrap(theta)
        if self.task == "balance" or (abs(theta) < .6 and abs(omega) < 3):
            force = -self.gain @ np.array([x, v, theta, omega])
        else:
            # Rod inertia about the pivot: (4/3)*m*l². Zero energy at upright.
            energy = .5 * ((4/3)*.1*.5**2) * omega**2 + .1*9.8*.5*(np.cos(theta)-1)
            force = np.clip(80 * energy * omega * np.cos(theta), -7, 7) - x - v
            if abs(omega) < .005 and abs(abs(theta)-np.pi) < .05:
                force += 1.  # break the exactly hanging equilibrium deterministically
        return np.array([np.clip(force / self.force_limit, -1, 1)], dtype=np.float32)
