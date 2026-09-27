"""Informe de verificación del entorno: versiones, constantes y convenciones de signo.

Uso:  python scripts/verify_environment.py

Pensado para ejecutarse al empezar cada sesión o en otra máquina: si algo de lo
que asumen los controladores (constantes, signos, integrador) no se cumple, el
script termina con código de salida 1 en lugar de seguir en silencio.
Además guarda results/figures/00_convencion_signos.png para las diapositivas.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # sin ventana: el script también debe correr en un servidor
import matplotlib.pyplot as plt
import numpy as np

from cartpole_lab import CartPoleTask, continuous_dynamics, load_params

FIGURE_PATH = Path(__file__).resolve().parents[1] / "results" / "figures" / "00_convencion_signos.png"


def sign_checks(params) -> dict[str, bool]:
    """Cada entrada es una afirmación física que LQR/MPC dan por cierta."""
    push_right = continuous_dynamics(np.zeros(4), +params.force_mag, params)
    tilted = continuous_dynamics([0.0, 0.0, 0.05, 0.0], 0.0, params)
    return {
        "F > 0 acelera el carro hacia +x (ẍ > 0)": push_right[1] > 0,
        "F > 0 hace girar el poste hacia θ < 0 (θ̈ < 0)": push_right[3] < 0,
        "Con F = 0, un θ > 0 crece (equilibrio inestable)": tilted[3] > 0,
        "El reposo vertical es un equilibrio (f(0, 0) = 0)": not np.any(continuous_dynamics(np.zeros(4), 0.0, params)),
    }


def save_sign_convention_figure() -> None:
    task = CartPoleTask(render_mode="rgb_array")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    try:
        for ax, theta in zip(axes, (-0.2, +0.2)):
            task.reset(seed=0, options={"initial_state": [0.0, 0.0, theta, 0.0]})
            ax.imshow(task.render()[120:330])
            ax.set_title(f"θ = {theta:+.1f} rad  →  poste hacia {'+x (derecha)' if theta > 0 else '−x (izquierda)'}")
            ax.axis("off")
    finally:
        task.close()
    fig.suptitle("Convención de signos de CartPole-v1 (Gymnasium 1.3.0): x y F positivos hacia la derecha")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def main() -> int:
    params = load_params()
    print(f"Python {platform.python_version()} | Gymnasium {gym.__version__} | NumPy {np.__version__}")
    print("\nConstantes leídas de CartPoleEnv (no copiadas a mano):")
    for name, value in vars(params).items():
        print(f"  {name:<24} = {value}")
    print(f"  {'total_mass':<24} = {params.total_mass}")
    print(f"  {'polemass_length':<24} = {params.polemass_length}")
    print(f"  θ límite = {np.degrees(params.theta_threshold_radians):.1f}°")

    checks = sign_checks(params)
    checks["Integrador del simulador = Euler explícito"] = params.kinematics_integrator == "euler"
    print("\nVerificaciones:")
    for claim, ok in checks.items():
        print(f"  [{'OK' if ok else 'FALLO'}] {claim}")

    save_sign_convention_figure()
    print(f"\nFigura guardada en {FIGURE_PATH}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
