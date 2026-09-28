r"""Actor-crítico A2C (Mnih et al., 2016) con ventaja GAE (Schulman et al., 2016) y acción continua.

IDEA
----
Dos redes con papeles distintos:
* ACTOR π_θ(F | s): la política. Aquí es CONTINUA: una gaussiana sobre una variable interna u,
  con F = F_max · tanh(u). Puede elegir cualquier fuerza en (−10, 10) N, como el LQR.
* CRÍTICO V_w(s): estima cuánto vale el estado (el retorno esperado desde s).

El crítico sirve para juzgar al actor mediante la VENTAJA: "¿esta acción salió mejor o peor de
lo que esperaba el crítico?". El actor sube la probabilidad de las acciones con ventaja
positiva:

    pérdida_actor  = − media[ log π_θ(F_t | s_t) · Â_t ]  − c_H · entropía
    pérdida_crítico =  ½ · media[ (V_w(s_t) − R̂_t)² ]

VENTAJA GAE(λ): media ponderada de estimaciones a 1, 2, 3... pasos del error TD
δ_t = r_t + γ V(s_{t+1}) − V(s_t):

    Â_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + …      (cortada al terminar el episodio)

λ = 0 da la ventaja TD de un paso (poca varianza, más sesgo); λ = 1, el retorno Monte Carlo
menos V (sin sesgo, mucha varianza). λ = 0.95 es el compromiso estándar (Schulman et al.).
Fin de episodio: fallo -> el futuro vale 0; truncamiento por tiempo -> se estima con V del
último estado REAL (guardado antes del reset), como en los pasos 8 y 9.

POR QUÉ tanh Y NO RECORTAR: si se muestrea F ~ N(μ, σ) y el actuador lo recorta a ±10 N, el
log π usado en el gradiente es el de la fuerza SIN recortar: gradiente sesgado. Con
F = F_max · tanh(u), toda muestra es admisible (el actuador nunca recorta) y log π se corrige
exactamente con el jacobiano del cambio de variable (como en SAC; Haarnoja et al., 2018):

    log π(F | s) = log N(u; μ, σ) − log( F_max · (1 − tanh² u) )

HIPERPARÁMETROS (los heurísticos se señalan)
-------------------------------------------
* Redes 4 → 64 → 64 → {1: μ del actor | 1: V del crítico}, ReLU, SEPARADAS: el actor y el
  crítico aprenden cosas distintas y no se estorban. log σ es un parámetro libre que no
  depende del estado: es la forma estándar en PPO y A2C continuos. HEURÍSTICO.
* 8 entornos en paralelo × 32 pasos = 256 transiciones por actualización (A2C síncrono):
  promedia el gradiente de la política, muy ruidoso con un solo episodio. HEURÍSTICO.
* γ = 0.99 (igual que el resto de RL), λ = 0.95 (valor estándar de GAE).
* Adam: lr 3·10⁻⁴ para el actor y 10⁻³ para el crítico (el crítico hace regresión y puede
  ir más rápido). Recorte del gradiente a norma 0.5. Coeficiente de entropía 10⁻³: evita
  que σ colapse antes de tiempo. Ventajas normalizadas por lote. Todos HEURÍSTICOS
  habituales (Andrychowicz et al., 2021).
* Última capa del actor inicializada con pesos ×0.01: μ ≈ 0 al empezar, así el único
  motor de la exploración inicial es σ (σ₀ = 1 en unidades de u). HEURÍSTICO estándar.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from cartpole_lab.env import CartPoleTask
from cartpole_lab.params import CartPoleParams, load_params
from cartpole_lab.rl.dqn import input_scales, network_forward_numpy, network_layers
from cartpole_lab.rollout import run_episode

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class ACConfig:
    total_steps: int = 307_200  # pasos de entorno SUMADOS entre entornos (= 1200 actualizaciones)
    n_envs: int = 8
    n_steps: int = 32
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 0.5
    initial_log_std: float = 0.0
    eval_every_steps: int = 10_240  # 40 actualizaciones
    n_eval_episodes: int = 10
    eval_seed_start: int = 2000  # CI de VALIDACIÓN, disjuntas del test (1000–1049)

    @property
    def batch(self) -> int:
        return self.n_envs * self.n_steps

    def __post_init__(self) -> None:
        checks = {
            "gamma": 0.0 <= self.gamma < 1.0,
            "gae_lambda": 0.0 <= self.gae_lambda <= 1.0,
            "n_envs": 1 <= self.n_envs <= 100,  # semillas de entorno seed·100 + i: sin colisiones
            "n_steps": self.n_steps >= 1,
            "lr": self.actor_lr > 0.0 and self.critic_lr > 0.0,
            "hidden": len(self.hidden) >= 1 and all(h >= 1 for h in self.hidden),
            "grad_clip_norm": self.grad_clip_norm > 0.0,
            "entropy_coef": self.entropy_coef >= 0.0,
        }
        if checks["n_envs"] and checks["n_steps"]:
            checks["total_steps"] = self.total_steps >= self.batch and self.total_steps % self.batch == 0
            checks["eval"] = (self.batch <= self.eval_every_steps <= self.total_steps
                              and self.eval_every_steps % self.batch == 0 and self.n_eval_episodes >= 1)
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"Configuración actor-crítico inválida en: {failed}")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


def _mlp(hidden: tuple[int, ...]) -> nn.Sequential:
    sizes = (4, *hidden)
    blocks: list[nn.Module] = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        blocks += [nn.Linear(n_in, n_out), nn.ReLU()]
    blocks.append(nn.Linear(sizes[-1], 1))
    return nn.Sequential(*blocks)


class GaussianActor(nn.Module):
    """μ_θ(s) (media de u) + log σ libre. La fuerza es F = F_max · tanh(u), u ~ N(μ, σ²)."""

    def __init__(self, hidden: tuple[int, ...], initial_log_std: float = 0.0) -> None:
        super().__init__()
        self.layers = _mlp(hidden)
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.01)  # μ ≈ 0 al empezar (ver docstring del módulo)
            self.layers[-1].bias.zero_()
        self.log_std = nn.Parameter(torch.tensor(float(initial_log_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class ValueCritic(nn.Module):
    def __init__(self, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.layers = _mlp(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def squashed_log_prob(u: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor, force_limit: float) -> torch.Tensor:
    """log π(F | s) para F = F_max·tanh(u), u ~ N(μ, σ²). Estable: log(1 − tanh² u) = 2(log 2 − u − softplus(−2u))."""
    gaussian = -0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * _LOG_2PI
    log_jacobian = math.log(force_limit) + 2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))
    return gaussian - log_jacobian


def next_state_values(v_next_obs: ArrayLike, v_final_obs: ArrayLike, terminated: ArrayLike, truncated: ArrayLike) -> np.ndarray:
    """Valor del estado siguiente para el error TD: 0 si FALLÓ, V(último estado real) si se TRUNCÓ."""
    return np.where(terminated, 0.0, np.where(truncated, v_final_obs, v_next_obs))


def gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, continues: np.ndarray,
        gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Ventajas GAE(λ) y retornos objetivo para el crítico. Arrays (T, N_entornos).

    `continues[t]` = 0 si el episodio acabó en t (por fallo o por tiempo): la suma no cruza episodios.
    """
    deltas = rewards + gamma * next_values - values
    advantages = np.zeros_like(deltas)
    running = np.zeros(deltas.shape[1:])
    for t in reversed(range(deltas.shape[0])):
        running = deltas[t] + gamma * lam * continues[t] * running
        advantages[t] = running
    return advantages, advantages + values


# --- Política exportable (NumPy, lo que ejecutará el navegador) ---------------------------------


class ACPolicy:
    """Política DETERMINISTA del actor: F = F_max · tanh(μ(s)). Controlador para run_episode."""

    action_mode = "continuous"

    def __init__(self, layers: list[dict[str, Any]], scales: tuple[float, ...], force_limit: float) -> None:
        self._layers = [{"W": np.asarray(l["W"], dtype=np.float32), "b": np.asarray(l["b"], dtype=np.float32),
                         "activation": l["activation"]} for l in layers]
        self._scales = np.asarray(scales, dtype=np.float64)
        self._force_limit = float(force_limit)

    def reset(self) -> None:
        pass

    def __call__(self, state: ArrayLike) -> float:
        s = np.asarray(state, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError(f"El estado debe tener forma (4,), no {s.shape}")
        mu = float(network_forward_numpy(self._layers, s / self._scales)[0])
        return self._force_limit * math.tanh(mu)


def export_actor(actor_or_layers: GaussianActor | list[dict[str, Any]], path: Path, *, input_scales: tuple[float, ...],
                 force_limit: float, metadata: dict[str, Any], log_std: float | None = None) -> None:
    if isinstance(actor_or_layers, GaussianActor):
        layers = network_layers(actor_or_layers)
        log_std = float(actor_or_layers.log_std) if log_std is None else log_std
    else:
        layers = actor_or_layers
        if log_std is None:  # sin el objeto de PyTorch no hay de dónde sacarlo: se exige explícitamente
            raise ValueError("Al exportar desde una lista de capas hay que pasar log_std")
    payload = {
        "format": "mlp-json-v1", "output": "tanh",
        "inference": "x = s / input_scales; para cada capa: x = W·x + b (+ ReLU salvo en la última); "
                     "F = force_limit · tanh(x[0])  (política determinista: la media, sin ruido)",
        "input_scales": list(input_scales), "force_limit": float(force_limit), "log_std": log_std,
        "layers": layers, "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


# --- Entrenamiento ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ACRun:
    config: ACConfig
    seed: int
    episode_returns: np.ndarray  # retornos de entrenamiento (política estocástica), en orden de fin
    eval_steps: np.ndarray
    eval_returns: np.ndarray  # (n_evaluaciones, n_eval_episodes) política DETERMINISTA en validación
    policy_std: np.ndarray  # σ (en unidades de u) tras cada actualización: ¿colapsa la exploración?
    explained_variance: np.ndarray  # del crítico en cada lote: ¿el crítico predice los retornos?
    final_layers: list[dict[str, Any]]
    final_log_std: float
    best_layers: list[dict[str, Any]]
    best_log_std: float  # log σ de la red guardada (la política determinista no lo usa; se documenta)
    best_eval_step: int
    best_eval_mean: float
    wall_time_s: float


class _Rollout:
    """Recoge n_steps × n_envs transiciones con la política estocástica actual."""

    def __init__(self, config: ACConfig, seed: int, scales: torch.Tensor, force_limit: float) -> None:
        self.config, self.scales, self.force_limit = config, scales, force_limit
        self.envs = [CartPoleTask(action_mode="continuous") for _ in range(config.n_envs)]
        self.obs = np.stack([env.reset(seed=seed * 100 + i)[0] for i, env in enumerate(self.envs)])
        self.running_returns = np.zeros(config.n_envs)
        self.finished_returns: list[float] = []

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def collect(self, actor: GaussianActor, critic: ValueCritic) -> dict[str, np.ndarray]:
        T, N = self.config.n_steps, self.config.n_envs
        data = {k: np.zeros((T, N)) for k in ("u", "values", "rewards", "final_values")}
        data.update(obs=np.zeros((T, N, 4)), terminated=np.zeros((T, N), bool), truncated=np.zeros((T, N), bool))
        for t in range(T):
            x = torch.as_tensor(self.obs, dtype=torch.float32) / self.scales
            with torch.no_grad():
                mu, values = actor(x), critic(x)
                u = mu + actor.log_std.exp() * torch.randn(N)
            data["obs"][t], data["u"][t], data["values"][t] = self.obs, u.numpy(), values.numpy()
            forces = self.force_limit * np.tanh(u.numpy())
            for i, env in enumerate(self.envs):
                self._step_env(i, env, float(forces[i]), t, data, critic)
        with torch.no_grad():
            data["last_values"] = critic(torch.as_tensor(self.obs, dtype=torch.float32) / self.scales).numpy()
        return data

    def _step_env(self, i: int, env: CartPoleTask, force: float, t: int, data: dict, critic: ValueCritic) -> None:
        obs, reward, terminated, truncated, _ = env.step(force)
        data["rewards"][t, i], data["terminated"][t, i], data["truncated"][t, i] = reward, terminated, truncated
        self.running_returns[i] += reward
        if truncated and not terminated:  # V del último estado REAL, antes del reset
            with torch.no_grad():
                data["final_values"][t, i] = float(critic(torch.as_tensor(obs, dtype=torch.float32) / self.scales))
        if terminated or truncated:
            self.finished_returns.append(float(self.running_returns[i]))
            self.running_returns[i] = 0.0
            obs, _ = env.reset()
        self.obs[i] = obs


def _advantages(data: dict[str, np.ndarray], config: ACConfig) -> tuple[np.ndarray, np.ndarray]:
    v_next_obs = np.concatenate([data["values"][1:], data["last_values"][None]], axis=0)
    next_values = next_state_values(v_next_obs, data["final_values"], data["terminated"], data["truncated"])
    continues = 1.0 - (data["terminated"] | data["truncated"])
    return gae(data["rewards"], data["values"], next_values, continues, config.gamma, config.gae_lambda)


def _update(actor: GaussianActor, critic: ValueCritic, optimizers: tuple, data: dict, config: ACConfig,
            scales: torch.Tensor, force_limit: float) -> float:
    """Una actualización A2C (un paso de gradiente por red). Devuelve la varianza explicada del crítico."""
    advantages, returns = _advantages(data, config)
    x = torch.as_tensor(data["obs"].reshape(-1, 4), dtype=torch.float32) / scales
    u = torch.as_tensor(data["u"].reshape(-1), dtype=torch.float32)
    adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalización por lote
    ret = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)

    log_prob = squashed_log_prob(u, actor(x), actor.log_std, force_limit)
    entropy = actor.log_std + 0.5 * (_LOG_2PI + 1.0)  # entropía de la gaussiana de u (aproximación estándar)
    actor_loss = -(log_prob * adv).mean() - config.entropy_coef * entropy
    critic_loss = 0.5 * ((critic(x) - ret) ** 2).mean()
    for loss, net, optimizer in ((actor_loss, actor, optimizers[0]), (critic_loss, critic, optimizers[1])):
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
        optimizer.step()
    residual = returns - data["values"]
    return float(1.0 - residual.var() / (returns.var() + 1e-8))


def _evaluate(layers: list[dict[str, Any]], scales_np: tuple[float, ...], force_limit: float, config: ACConfig,
              task: CartPoleTask) -> np.ndarray:
    policy = ACPolicy(layers, scales_np, force_limit)
    seeds = range(config.eval_seed_start, config.eval_seed_start + config.n_eval_episodes)
    return np.array([run_episode(task, policy, seed=s).total_reward for s in seeds])


def train_a2c(config: ACConfig, seed: int, params: CartPoleParams | None = None) -> ACRun:
    """Entrena desde cero. Reproducible en CPU (semillas de PyTorch y de los entornos; 1 hilo)."""
    params = params or load_params()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    scales_np = input_scales(params)
    scales = torch.as_tensor(scales_np, dtype=torch.float32)
    actor, critic = GaussianActor(config.hidden, config.initial_log_std), ValueCritic(config.hidden)
    optimizers = (torch.optim.Adam(actor.parameters(), lr=config.actor_lr),
                  torch.optim.Adam(critic.parameters(), lr=config.critic_lr))
    rollout, eval_task = _Rollout(config, seed, scales, params.force_mag), CartPoleTask(action_mode="continuous")
    log: dict[str, list] = {"std": [], "ev": [], "eval_steps": [], "eval_returns": []}
    best: dict[str, Any] = {"layers": network_layers(actor), "log_std": float(actor.log_std), "step": 0, "mean": -np.inf}
    start = time.perf_counter()
    try:
        for update in range(config.total_steps // config.batch):
            data = rollout.collect(actor, critic)
            log["ev"].append(_update(actor, critic, optimizers, data, config, scales, params.force_mag))
            log["std"].append(float(actor.log_std.exp()))
            steps = (update + 1) * config.batch
            if steps % config.eval_every_steps == 0:
                layers = network_layers(actor)
                returns = _evaluate(layers, scales_np, params.force_mag, config, eval_task)
                log["eval_steps"].append(steps)
                log["eval_returns"].append(returns)
                if returns.mean() > best["mean"]:  # estricto: ante empate se queda la primera
                    best = {"layers": layers, "log_std": float(actor.log_std), "step": steps, "mean": float(returns.mean())}
    finally:
        rollout.close()
        eval_task.close()
    arrays = {"episode_returns": np.array(rollout.finished_returns), "eval_steps": np.array(log["eval_steps"]),
              "eval_returns": np.array(log["eval_returns"]), "policy_std": np.array(log["std"]),
              "explained_variance": np.array(log["ev"])}
    for array in arrays.values():
        array.setflags(write=False)
    return ACRun(config=config, seed=seed, final_layers=network_layers(actor), final_log_std=float(actor.log_std),
                 best_layers=best["layers"], best_log_std=best["log_std"], best_eval_step=best["step"],
                 best_eval_mean=best["mean"],
                 wall_time_s=time.perf_counter() - start, **arrays)
