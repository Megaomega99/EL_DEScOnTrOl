r"""Deep Q-Network (Mnih et al., 2015): Q-learning con una red neuronal en lugar de una tabla.

POR QUÉ UNA RED
---------------
La tabla del paso 8 agrupa estados distintos en la misma caja, y pierde información: el
aprendizaje oscila y hay cajas que nunca se aprenden. Una red Q_φ(s, ·) recibe el estado
CONTINUO y devuelve un valor por cada acción, interpolando entre estados parecidos.

La actualización es la de Q-learning, convertida en regresión:

    pérdida = Huber( Q_φ(s, a),  r + γ · (1 − terminado) · max_a' Q_φ⁻(s', a') )

Dos piezas la hacen estable (sin ellas, Q-learning con redes suele divergir):
* Memoria de repetición (replay.py): se entrena con lotes aleatorios de transiciones
  pasadas, no con la secuencia correlacionada del episodio en curso.
* Red objetivo Q_φ⁻: una copia congelada de la red, sincronizada cada `target_update_every`
  pasos. Sin ella, el objetivo se mueve con cada actualización ("perseguirse la cola").

HIPERPARÁMETROS Y JUSTIFICACIÓN (lo que no tiene justificación teórica se dice)
------------------------------------------------------------------------------
* Arquitectura 4 → 64 → 64 → 5, ReLU. SUPUESTO HEURÍSTICO: dos capas ocultas bastan para
  aproximar funciones suaves de 4 variables. 64 unidades es un tamaño habitual en tareas
  de control de baja dimensión, y deja 4.8 k parámetros, fáciles de ejecutar en el navegador.
* Entradas normalizadas: x / 2.4 m y θ / 12° (los límites de fallo, con base física);
  ẋ / 1.0 m/s y θ̇ / 1.5 rad/s (el orden de magnitud observado: el percentil 99 con una
  política aleatoria es 1.2 m/s y 1.7 rad/s). Objetivo: entradas O(1) para que el
  optimizador trate las 4 variables por igual.
* γ = 0.99: el mismo que en el paso 8 (horizonte de 2 s), para comparar métodos.
* Huber en lugar de error cuadrático, como en Mnih et al. (2015): acota el gradiente de los
  errores grandes, frecuentes al principio, cuando los objetivos están muy lejos.
* Adam con lr = 1e-3 (el valor por defecto de Adam) y lote de 64. Recorte del gradiente a
  norma 10: protección frente a gradientes explosivos. Los valores son HEURÍSTICOS.
* Memoria de 50 000 transiciones (~100 episodios buenos de 500 pasos). Aprendizaje a partir
  de 1 000 transiciones, para que los primeros lotes no sean casi idénticos.
* Red objetivo sincronizada cada 500 pasos, un episodio completo de CartPole-v1. HEURÍSTICO.
* ε lineal de 1.0 a 0.05 en el primer 20 % de los pasos. HEURÍSTICO.
* Terminado frente a truncado: igual que en el paso 8, solo el fallo real anula el bootstrap.
* Presupuesto en PASOS de entorno (no en episodios): es lo que cuesta la experiencia.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from cartpole_lab.env import CartPoleTask
from cartpole_lab.params import CartPoleParams, load_params
from cartpole_lab.rl.replay import Batch, ReplayBuffer
from cartpole_lab.rl.tabular import FORCE_LEVELS
from cartpole_lab.rollout import run_episode

VELOCITY_SCALES = (1.0, 1.5)  # ẋ [m/s], θ̇ [rad/s]: orden de magnitud observado (docstring)


def input_scales(params: CartPoleParams) -> tuple[float, float, float, float]:
    return (params.x_threshold, VELOCITY_SCALES[0], params.theta_threshold_radians, VELOCITY_SCALES[1])


@dataclass(frozen=True)
class DQNConfig:
    total_steps: int = 100_000
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 50_000
    learning_starts: int = 1_000
    target_update_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.2
    grad_clip_norm: float = 10.0
    eval_every_steps: int = 5_000
    n_eval_episodes: int = 10
    eval_seed_start: int = 2000  # CI de VALIDACIÓN, disjuntas del test (1000–1049)
    force_levels: tuple[float, ...] = field(default=FORCE_LEVELS)
    double: bool = False  # Double DQN (van Hasselt et al., 2016): ver dqn_targets

    def __post_init__(self) -> None:
        checks = {
            "gamma": 0.0 <= self.gamma < 1.0,
            "learning_rate": self.learning_rate > 0.0,
            "batch_size": self.batch_size >= 1,
            "buffer_capacity": self.buffer_capacity >= self.batch_size,
            "learning_starts": self.batch_size <= self.learning_starts < self.total_steps,
            "epsilon": 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0,
            "epsilon_decay_fraction": 0.0 < self.epsilon_decay_fraction <= 1.0,
            "hidden": len(self.hidden) >= 1 and all(h >= 1 for h in self.hidden),
            "eval": 1 <= self.eval_every_steps <= self.total_steps and self.n_eval_episodes >= 1,
            "target_update_every": self.target_update_every >= 1,
            "grad_clip_norm": self.grad_clip_norm > 0.0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"Configuración DQN inválida en: {failed}")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def epsilon_at(self, step: int) -> float:
        progress = min(1.0, step / max(1, self.epsilon_decay_fraction * self.total_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


class QNetwork(nn.Module):
    def __init__(self, hidden: tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        sizes = (4, *hidden)
        blocks: list[nn.Module] = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            blocks += [nn.Linear(n_in, n_out), nn.ReLU()]
        blocks.append(nn.Linear(sizes[-1], n_actions))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def q_upper_bound(gamma: float) -> float:
    """Máximo valor Q alcanzable con recompensa +1 por paso: Σ γ^k = 1/(1−γ) (100 con γ = 0.99).

    Como el truncamiento por tiempo se trata con bootstrap, el horizonte efectivo es infinito.
    Una red que estima Q por encima de esta cota está SOBREESTIMANDO, sin ambigüedad.
    """
    return 1.0 / (1.0 - gamma)


def dqn_targets(
    rewards: torch.Tensor, next_q_target: torch.Tensor, terminated: torch.Tensor, gamma: float,
    next_q_online: torch.Tensor | None = None,
) -> torch.Tensor:
    """Objetivo TD; sin futuro si el episodio TERMINÓ (fallo real).

    * DQN        : r + γ · max_a' Q⁻(s', a')
    * Double DQN : r + γ · Q⁻(s', argmax_a' Q(s', a'))  (se pasa `next_q_online`)
    Por qué Double: el max de estimaciones ruidosas es sesgado al alza (el ruido positivo
    siempre "gana"). Si una red elige la acción y OTRA la evalúa, el sesgo desaparece en
    media. Motivación empírica en este proyecto: el DQN estimó Q(s=0) ≈ 109 > 100, la cota
    física (docs/09_dqn.md).
    """
    if next_q_online is None:
        bootstrap = next_q_target.max(dim=1).values
    else:
        bootstrap = next_q_target.gather(1, next_q_online.argmax(dim=1, keepdim=True)).squeeze(1)
    return rewards + gamma * (~terminated).float() * bootstrap


# --- Exportación e inferencia sin PyTorch (lo que ejecutará el navegador) ---------------------------


def network_layers(net: QNetwork) -> list[dict[str, Any]]:
    linears = [m for m in net.layers if isinstance(m, nn.Linear)]
    return [
        {"W": lin.weight.detach().cpu().numpy().astype(np.float32).tolist(),  # (salidas, entradas): y = W x + b
         "b": lin.bias.detach().cpu().numpy().astype(np.float32).tolist(),
         "activation": "relu" if k < len(linears) - 1 else "linear"}
        for k, lin in enumerate(linears)
    ]


def network_forward_numpy(layers: list[dict[str, Any]], x_normalized: ArrayLike) -> np.ndarray:
    """Pasada hacia delante en float32 con NumPy: la referencia para la versión en JavaScript."""
    x = np.asarray(x_normalized, dtype=np.float32)
    for layer in layers:
        x = x @ np.asarray(layer["W"], dtype=np.float32).T + np.asarray(layer["b"], dtype=np.float32)
        if layer["activation"] == "relu":
            x = np.maximum(x, 0.0)
    return x


def export_network(
    net_or_layers: QNetwork | list[dict[str, Any]], path: Path, *, input_scales: tuple[float, ...],
    force_levels: tuple[float, ...], metadata: dict[str, Any],
) -> None:
    layers = network_layers(net_or_layers) if isinstance(net_or_layers, QNetwork) else net_or_layers
    payload = {
        "format": "mlp-json-v1",
        "inference": "x = s / input_scales; para cada capa: x = W·x + b (+ ReLU salvo en la última); "
                     "acción = force_levels[argmax(x)]",
        "input_scales": list(input_scales), "force_levels": list(force_levels), "layers": layers,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def load_network(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class DQNPolicy:
    """Política greedy de una red exportada (controlador para run_episode). Usa solo NumPy."""

    action_mode = "continuous"

    def __init__(self, layers: list[dict[str, Any]], scales: tuple[float, ...], force_levels: tuple[float, ...]) -> None:
        self._layers = [{"W": np.asarray(l["W"], dtype=np.float32), "b": np.asarray(l["b"], dtype=np.float32),
                         "activation": l["activation"]} for l in layers]
        self._scales = np.asarray(scales, dtype=np.float64)
        self._forces = np.asarray(force_levels, dtype=np.float64)

    def reset(self) -> None:
        pass

    def __call__(self, state: ArrayLike) -> float:
        s = np.asarray(state, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError(f"El estado debe tener forma (4,), no {s.shape}")
        q = network_forward_numpy(self._layers, s / self._scales)
        return float(self._forces[int(np.argmax(q))])


# --- Entrenamiento ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DQNRun:
    config: DQNConfig
    seed: int
    episode_returns: np.ndarray  # (E,) retorno de cada episodio de entrenamiento (con exploración)
    episode_end_steps: np.ndarray  # (E,) paso de entorno en que terminó cada episodio
    eval_steps: np.ndarray  # pasos tras los que se evaluó
    eval_episodes: np.ndarray  # episodios completados en ese momento
    eval_returns: np.ndarray  # (n_evaluaciones, n_eval_episodes) retorno greedy en validación
    mean_losses: np.ndarray  # pérdida media en cada intervalo entre evaluaciones (diagnóstico)
    q_at_rest: np.ndarray  # max_a Q(s = 0, a) en cada evaluación: diagnóstico de sobreestimación
    action_gap_median: np.ndarray  # mediana de Q(mejor) − Q(segunda) en PROBE_STATES
    q_max_median: np.ndarray  # mediana de max_a Q en PROBE_STATES (para dar escala al gap)
    final_layers: list[dict[str, Any]]
    best_layers: list[dict[str, Any]]  # mejor red vista en VALIDACIÓN (parada temprana)
    best_eval_step: int
    best_eval_mean: float
    wall_time_s: float


def _probe_states() -> np.ndarray:
    """256 estados fijos cerca del equilibrio, para seguir la "diferencia entre acciones" (action gap).

    Se generan con una semilla PROPIA (4242), independiente del entrenamiento. Caja:
    ±(0.5 m, 0.5 m/s, 0.1 rad, 0.5 rad/s): donde vive una política que funciona.
    """
    return np.random.default_rng(4242).uniform(-1, 1, size=(256, 4)) * np.array([0.5, 0.5, 0.1, 0.5])


PROBE_STATES = _probe_states()


def _evaluate(layers: list[dict[str, Any]], scales: tuple[float, ...], config: DQNConfig, task: CartPoleTask) -> np.ndarray:
    policy = DQNPolicy(layers, scales, config.force_levels)
    seeds = range(config.eval_seed_start, config.eval_seed_start + config.n_eval_episodes)
    return np.array([run_episode(task, policy, seed=s).total_reward for s in seeds])


def value_diagnostics(q_net: QNetwork, scales: torch.Tensor) -> dict[str, float]:
    """Diagnóstico del porqué del colapso (docs/09_dqn.md):
    * q_at_rest: max_a Q(0, a), comparable con la cota física 1/(1−γ) = 100 (sobreestimación).
    * gap_median: mediana, sobre PROBE_STATES, de Q(mejor acción) − Q(segunda). Es lo único que
      decide la política; si es diminuto frente a Q, el error de la red basta para cambiarla.
    """
    with torch.no_grad():
        q_rest = float(q_net(torch.zeros(1, 4)).max())
        q = q_net(torch.as_tensor(PROBE_STATES, dtype=torch.float32) / scales)
        top2 = torch.topk(q, 2, dim=1).values
    gaps = (top2[:, 0] - top2[:, 1]).numpy()
    return {"q_at_rest": q_rest, "gap_median": float(np.median(gaps)),
            "q_max_median": float(np.median(top2[:, 0].numpy()))}


def _gradient_step(
    q_net: QNetwork, target_net: QNetwork, optimizer: torch.optim.Optimizer, batch: Batch,
    config: DQNConfig, scales: torch.Tensor,
) -> float:
    states = torch.as_tensor(batch.states) / scales
    next_states = torch.as_tensor(batch.next_states) / scales
    actions = torch.as_tensor(batch.actions)
    q = q_net(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        target = dqn_targets(torch.as_tensor(batch.rewards), target_net(next_states),
                             torch.as_tensor(batch.terminated), config.gamma,
                             next_q_online=q_net(next_states) if config.double else None)
    loss = nn.functional.smooth_l1_loss(q, target)  # Huber
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), config.grad_clip_norm)
    optimizer.step()
    return float(loss)


class _Learner:
    """Estado mutable del entrenamiento (redes, optimizador, memoria, RNG). Encapsulado para que
    `train_dqn` sea solo el bucle que orquesta: actuar -> guardar -> aprender -> sincronizar -> evaluar."""

    def __init__(self, config: DQNConfig, seed: int, scales: torch.Tensor) -> None:
        torch.manual_seed(seed)
        self.config, self.scales = config, scales
        self.rng = np.random.default_rng(seed)
        self.q_net = QNetwork(config.hidden, len(config.force_levels))
        self.target_net = QNetwork(config.hidden, len(config.force_levels))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_capacity)

    def act(self, state: np.ndarray, step: int) -> int:
        """ε-greedy."""
        if self.rng.random() < self.config.epsilon_at(step):
            return int(self.rng.integers(len(self.config.force_levels)))
        with torch.no_grad():
            return int(self.q_net(torch.as_tensor(state, dtype=torch.float32) / self.scales).argmax())

    def learn(self, step: int) -> float | None:
        """Un paso de gradiente por paso de entorno a partir de `learning_starts`; sincroniza la red objetivo."""
        loss = None
        if step + 1 >= self.config.learning_starts:
            batch = self.buffer.sample(self.config.batch_size, self.rng)
            loss = _gradient_step(self.q_net, self.target_net, self.optimizer, batch, self.config, self.scales)
        if (step + 1) % self.config.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss


def _checkpoint(learner: _Learner, step: int, eval_task: CartPoleTask, scales_np: tuple[float, ...],
                log: dict[str, list], interval_losses: list[float], best: dict[str, Any]) -> dict[str, Any]:
    """Evaluación greedy en validación + diagnósticos + mejor red vista. Devuelve el nuevo `best`."""
    layers = network_layers(learner.q_net)
    returns = _evaluate(layers, scales_np, learner.config, eval_task)
    log["eval_steps"].append(step + 1)
    log["eval_eps"].append(len(log["returns"]))
    log["eval_returns"].append(returns)
    log["losses"].append(float(np.mean(interval_losses)) if interval_losses else float("nan"))
    for key, value in value_diagnostics(learner.q_net, learner.scales).items():
        log[key].append(value)
    if returns.mean() > best["mean"]:  # estricto: ante empate se queda la primera
        return {"layers": layers, "step": step + 1, "mean": float(returns.mean())}
    return best


def train_dqn(config: DQNConfig, seed: int, params: CartPoleParams | None = None) -> DQNRun:
    """Entrena desde cero. Reproducible en CPU: siembra NumPy, PyTorch y el entorno; 1 hilo."""
    params = params or load_params()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    scales_np = input_scales(params)
    learner = _Learner(config, seed, torch.as_tensor(scales_np, dtype=torch.float32))
    task, eval_task = CartPoleTask(action_mode="continuous"), CartPoleTask(action_mode="continuous")
    log: dict[str, list] = {k: [] for k in ("returns", "ends", "eval_steps", "eval_eps", "eval_returns", "losses",
                                             "q_at_rest", "gap_median", "q_max_median")}
    best: dict[str, Any] = {"layers": network_layers(learner.q_net), "step": 0, "mean": -np.inf}
    interval_losses: list[float] = []
    start = time.perf_counter()
    try:
        state, _ = task.reset(seed=seed)
        episode_return = 0.0
        for step in range(config.total_steps):
            action = learner.act(state, step)
            next_state, reward, terminated, truncated, _ = task.step(config.force_levels[action])
            learner.buffer.add(state, action, reward, next_state, terminated)  # next_state REAL, antes del reset
            state, episode_return = next_state, episode_return + reward
            if terminated or truncated:
                log["returns"].append(episode_return)
                log["ends"].append(step + 1)
                state, _ = task.reset()
                episode_return = 0.0
            loss = learner.learn(step)
            if loss is not None:
                interval_losses.append(loss)
            if (step + 1) % config.eval_every_steps == 0:
                best = _checkpoint(learner, step, eval_task, scales_np, log, interval_losses, best)
                interval_losses = []
    finally:
        task.close()
        eval_task.close()
    return _build_run(config, seed, learner, log, best, time.perf_counter() - start)


def _build_run(config: DQNConfig, seed: int, learner: _Learner, log: dict[str, list], best: dict[str, Any],
               wall_time: float) -> DQNRun:
    names = {"episode_returns": "returns", "episode_end_steps": "ends", "eval_steps": "eval_steps",
             "eval_episodes": "eval_eps", "eval_returns": "eval_returns", "mean_losses": "losses",
             "q_at_rest": "q_at_rest", "action_gap_median": "gap_median", "q_max_median": "q_max_median"}
    arrays = {field_name: np.array(log[key]) for field_name, key in names.items()}
    for array in arrays.values():
        array.setflags(write=False)
    return DQNRun(config=config, seed=seed, final_layers=network_layers(learner.q_net), best_layers=best["layers"],
                  best_eval_step=best["step"], best_eval_mean=best["mean"], wall_time_s=wall_time, **arrays)
