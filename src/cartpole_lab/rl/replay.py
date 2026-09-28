"""Memoria de repetición (experience replay) para DQN.

Por qué existe: entrenar una red con transiciones CONSECUTIVAS es entrenarla con datos muy
correlacionados (el estado cambia poco de un paso al siguiente), y el descenso de gradiente
supone datos aproximadamente independientes. Guardar muchas transiciones y muestrear lotes al
azar rompe esa correlación y reutiliza cada experiencia varias veces (Mnih et al., 2015).

Implementación: buffer circular de tamaño fijo en arrays de NumPy; al llenarse, cada
transición nueva sustituye a la más antigua.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Batch:
    """Lote de transiciones muestreadas (copias: modificarlo no altera la memoria)."""

    states: np.ndarray  # (B, 4) float32
    actions: np.ndarray  # (B,) int64
    rewards: np.ndarray  # (B,) float32
    next_states: np.ndarray  # (B, 4) float32
    terminated: np.ndarray  # (B,) bool. SOLO fallo real; el truncamiento por tiempo NO se marca


class ReplayBuffer:
    """Buffer circular de transiciones (s, a, r, s', terminado); ver el docstring del módulo."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"La capacidad debe ser >= 1, no {capacity}")
        self.capacity = capacity
        self._states = np.zeros((capacity, 4), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, 4), dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=bool)
        self._next = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, state: ArrayLike, action: int, reward: float, next_state: ArrayLike, terminated: bool) -> None:
        i = self._next
        self._states[i], self._actions[i], self._rewards[i] = state, action, reward
        self._next_states[i], self._terminated[i] = next_state, terminated
        self._next = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Muestreo uniforme CON reemplazo entre las transiciones guardadas."""
        if self._size == 0:
            raise ValueError("No se puede muestrear una memoria vacía")
        idx = rng.integers(0, self._size, size=batch_size)
        return Batch(self._states[idx], self._actions[idx], self._rewards[idx], self._next_states[idx], self._terminated[idx])
