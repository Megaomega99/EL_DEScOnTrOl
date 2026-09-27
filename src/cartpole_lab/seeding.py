"""Semillas globales para reproducibilidad total entre corridas.

Qué cubre y qué NO:
* Python `random`, NumPy global (`np.random.*`) y PyTorch (CPU y CUDA) si está instalado.
* El entorno NO se siembra aquí: Gymnasium usa su propio generador, que se siembra
  con `task.reset(seed=...)`. Llamar a reset(seed) solo en el primer episodio basta
  para que la secuencia de condiciones iniciales sea reproducible; para evaluar
  sobre N semillas independientes, pasar una semilla distinta en cada reset.
* El determinismo de operaciones de PyTorch en GPU (cuDNN) se configura en la Fase 2.
"""

from __future__ import annotations

import numbers
import random

import numpy as np

_MAX_SEED = 2**32  # límite de np.random.seed


def set_global_seeds(seed: int) -> np.random.Generator:
    """Siembra todos los generadores globales y devuelve un `Generator` de NumPy nuevo."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"La semilla debe ser un entero, no {seed!r}")
    if not 0 <= seed < _MAX_SEED:
        raise ValueError(f"La semilla debe estar en [0, 2**32), no {seed}")

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # PyTorch solo hace falta en la Fase 2 (DQN, actor-crítico)
        pass
    else:
        torch.manual_seed(seed)  # siembra también todas las GPUs
    return np.random.default_rng(seed)
