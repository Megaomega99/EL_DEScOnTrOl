r"""Modelo dinámico explícito de CartPole-v1 — el "modelo interno" de LQR y MPC.

QUÉ MODELO ES (y cuál NO)
-------------------------
Estas funciones transcriben, operación por operación, `CartPoleEnv.step()` de
Gymnasium 1.3.0. No se deriva nada desde cero: el test
`test_euler_step_reproduces_gymnasium_step` exige igualdad EXACTA (bit a bit)
con el simulador para estados y fuerzas aleatorios.

Estado s = [x, ẋ, θ, θ̇]:  x posición del carro [m] (+ a la derecha),
θ ángulo del poste respecto a la vertical [rad] (+ = poste inclinado hacia +x).

1) Tiempo continuo, ṡ = f(s, F). Ecuaciones del carro-péndulo SIN rozamiento
   (Gymnasium remite a Florian 2007, https://coneural.org/florian/papers/05_cart_pole.pdf,
   y su implementación no incluye términos de fricción):

       temp = (F + m l θ̇² sin θ) / (M + m)
       θ̈    = (g sin θ − cos θ · temp) / ( l · (4/3 − m cos² θ / (M + m)) )
       ẍ    = temp − m l θ̈ cos θ / (M + m)

   El factor 4/3 sale de modelar el poste como una varilla uniforme de longitud
   2l: su inercia respecto al pivote es I = m(2l)²/12 + m l² = (4/3) m l².

2) Tiempo discreto: Gymnasium NO integra la EDO de forma exacta. Usa Euler
   explícito con Δt = τ = 0.02 s y fuerza constante durante el paso (ZOH):

       s_{k+1} = s_k + τ · f(s_k, F_k)

   Las posiciones se actualizan con las velocidades VIEJAS (no es Euler
   semi-implícito). Por tanto la planta "real" que ve cualquier controlador es
   este mapa discreto, no la EDO continua. Decisión del proyecto: los
   controladores basados en modelo usan ESTE mapa (y su linealización
   A_d = I + τ A_c, B_d = τ B_c en el paso 4), porque es exactamente lo que se
   simula. La discretización exacta de la EDO (A_d = e^{A_c τ}) difiere en
   O(τ²); esa diferencia se cuantifica al implementar LQR.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cartpole_lab.params import CartPoleParams

STATE_DIM = 4
STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")
STATE_UNITS = ("m", "m/s", "rad", "rad/s")


def _as_state(state: ArrayLike) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64)
    if s.shape[-1:] != (STATE_DIM,):
        raise ValueError(f"El estado debe tener última dimensión {STATE_DIM}, forma recibida {s.shape}")
    return s


def continuous_dynamics(state: ArrayLike, force: ArrayLike, params: CartPoleParams) -> np.ndarray:
    """ṡ = f(s, F). Admite lotes: `state` de forma (..., 4) y `force` de forma (...)."""
    s = _as_state(state)
    force = np.asarray(force, dtype=np.float64)
    x_dot, theta, theta_dot = s[..., 1], s[..., 2], s[..., 3]
    p = params

    # Mismo orden de operaciones que Gymnasium para obtener el mismo resultado en float64.
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + p.polemass_length * np.square(theta_dot) * sintheta) / p.total_mass
    thetaacc = (p.gravity * sintheta - costheta * temp) / (
        p.length * (4.0 / 3.0 - p.masspole * np.square(costheta) / p.total_mass)
    )
    xacc = temp - p.polemass_length * thetaacc * costheta / p.total_mass

    # broadcast_to: si llega un lote de fuerzas con un único estado, las velocidades
    # deben repetirse para que las cuatro componentes tengan la misma forma.
    x_dot = np.broadcast_to(x_dot, xacc.shape)
    theta_dot = np.broadcast_to(theta_dot, thetaacc.shape)
    return np.stack([x_dot, xacc, theta_dot, thetaacc], axis=-1)


def euler_step(state: ArrayLike, force: ArrayLike, params: CartPoleParams) -> np.ndarray:
    """s_{k+1} = s_k + τ f(s_k, F_k): el mapa discreto que ejecuta CartPole-v1."""
    if params.kinematics_integrator != "euler":
        # Fallo explícito: si alguien cambia el integrador del simulador, este modelo
        # dejaría de coincidir con él y los controladores basados en modelo mentirían.
        raise ValueError(
            f"El modelo solo replica el integrador 'euler'; el entorno usa "
            f"{params.kinematics_integrator!r}"
        )
    s = _as_state(state)
    return s + params.tau * continuous_dynamics(s, force, params)
