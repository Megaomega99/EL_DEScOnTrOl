r"""Discretización del estado continuo en "cajas" (paso 7), para Q-learning y SARSA tabulares.

POR QUÉ HACE FALTA
------------------
Un método tabular guarda un valor Q(s, a) por cada par (estado, acción). Con un estado
continuo de 4 dimensiones hay infinitos estados, así que se agrupan en un número finito
de cajas. Cada variable se corta en intervalos, y la caja es la combinación de intervalos.

PUNTO DE PARTIDA: LAS 162 CAJAS DE BARTO, SUTTON Y ANDERSON (1983)
------------------------------------------------------------------
Es la partición del artículo en el que se basa el CartPole de Gymnasium. Umbrales
verificados en el código original `pole.c` (función get_box; Gymnasium lo cita como fuente):

    x      : −0.8, 0.8 m                    -> 3 intervalos  (¿cerca de un borde del raíl?)
    ẋ      : −0.5, 0.5 m/s                  -> 3             (¿se mueve deprisa?)
    θ      : −6°, −1°, 0, 1°, 6°            -> 6             (fino cerca de la vertical)
    θ̇      : −50°/s, 50°/s                  -> 3
                                               3·3·6·3 = 162 cajas

Lógica de diseño: resolución FINA donde el control necesita precisión (θ cerca de 0) y
GRUESA donde basta saber "hacia qué lado" (posición y velocidades). Con 162 cajas × 5
acciones hay 810 valores que aprender: pocos, así que se aprende rápido. Con el algoritmo
original (un actor-crítico), `pole.c` reporta que equilibró el poste más de 100 000
pasos tras 79 ensayos.

Detalle: pole.c define los grados truncados (one_degree = 0.0174532). Aquí se usan
radianes exactos; la diferencia (~7·10⁻⁸ rad) es irrelevante.

JUSTIFICACIÓN DEL NÚMERO DE CAJAS
---------------------------------
No se asume: se mide. Se compara BOXES con una versión más gruesa y otra más fina,
con los mismos hiperparámetros y 10 semillas (scripts/train_tabular.py). El criterio se
declaró ANTES de ver resultados: mayor retorno greedy final (media sobre semillas, CI de
validación); si dos quedan a menos de una desviación típica, gana la de menos estados.
    "sin_carro": 1·1·6·3 = 18    (ignora x y ẋ: ¿hace falta ver el carro?)
    "boxes_1983": 162
    "fina"     : 3·3·8·5 = 360   (doble resolución en las variables del poste)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class BoxDiscretizer:
    """Cajas producto: `edges[v]` son los bordes INTERIORES de la variable v (vacío = 1 intervalo).

    Convención de pole.c: un valor igual a un borde pertenece a la caja SUPERIOR. Los valores
    fuera del rango caen en las cajas extremas; el fallo lo señala el entorno, no una caja
    especial.
    """

    edges: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.edges) != 4:
            raise ValueError(f"Se necesitan bordes para las 4 variables, se recibieron {len(self.edges)}")
        for variable, e in enumerate(self.edges):
            values = np.asarray(e, dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
                raise ValueError(f"Los bordes de la variable {variable} deben ser finitos y estrictamente crecientes")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(e) + 1 for e in self.edges)

    @property
    def n_states(self) -> int:
        return int(np.prod(self.shape))

    def box(self, state: ArrayLike) -> tuple[int, ...]:
        s = np.asarray(state, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError(f"El estado debe tener forma (4,), no {s.shape}")
        return tuple(int(np.searchsorted(e, value, side="right")) for e, value in zip(self.edges, s))

    def index(self, state: ArrayLike) -> int:
        """Índice plano 0..n_states−1: la fila de la tabla Q."""
        return int(np.ravel_multi_index(self.box(state), self.shape))


def _deg(*values: float) -> tuple[float, ...]:
    return tuple(float(np.radians(v)) for v in values)


BOXES_1983 = BoxDiscretizer(
    edges=((-0.8, 0.8), (-0.5, 0.5), _deg(-6.0, -1.0, 0.0, 1.0, 6.0), _deg(-50.0, 50.0))
)

DISCRETIZATIONS = {
    "sin_carro": BoxDiscretizer(edges=((), (), BOXES_1983.edges[2], BOXES_1983.edges[3])),
    "boxes_1983": BOXES_1983,
    "fina": BoxDiscretizer(
        edges=((-0.8, 0.8), (-0.5, 0.5), _deg(-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0), _deg(-50.0, -15.0, 15.0, 50.0))
    ),
}
