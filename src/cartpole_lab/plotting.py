"""Estilo común de las figuras de la charla (matplotlib).

Paleta: instancia de referencia validada para daltonismo (orden categorical fijo,
nunca cíclico). Se usan los mismos colores en todas las figuras para que cada
método conserve su color a lo largo de la presentación.
"""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
MUTED = "#a09f98"  # referencias (límites de fallo), nunca datos

# Orden fijo: el slot 1 es siempre la serie principal de cada figura.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# El color sigue al MÉTODO en toda la charla (nunca a su posición en una figura concreta):
# ocho métodos, ocho slots, en el orden en que se presentan.
METHOD_COLORS = {
    "PID": SERIES[0],
    "LQR": SERIES[1],
    "MPC": SERIES[2],
    "Fuzzy": SERIES[3],
    "Q-learning": SERIES[4],
    "SARSA": SERIES[5],
    "DQN": SERIES[6],
    "Actor-crítico": SERIES[7],
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_PRIMARY,
            "font.size": 11,
        }
    )
