"""cartpole_lab: control clásico vs. aprendizaje por refuerzo sobre CartPole-v1.

Módulos:
    params        constantes físicas leídas de Gymnasium
    dynamics      modelo explícito (EDO continua + paso de Euler idéntico al simulador)
    env           CartPoleTask, la interfaz común para todos los métodos
    cost          coste de etapa común para evaluación
    disturbances  perturbaciones externas (pulsos de fuerza sobre el carro)
    rollout       bucle de simulación común y registro de trayectorias
    seeding       semillas globales
    controllers   PID, LQR, MPC, fuzzy (Fase 1, pasos 3-6)
    rl            Q-learning, SARSA, DQN, actor-crítico (Fase 2)
"""

from cartpole_lab.cost import QuadraticCost
from cartpole_lab.disturbances import ForcePulse
from cartpole_lab.dynamics import continuous_dynamics, euler_step
from cartpole_lab.env import CartPoleTask
from cartpole_lab.params import ENV_ID, CartPoleParams, load_params
from cartpole_lab.rollout import Controller, Trajectory, run_episode
from cartpole_lab.seeding import set_global_seeds

__all__ = [
    "ENV_ID",
    "CartPoleParams",
    "CartPoleTask",
    "Controller",
    "ForcePulse",
    "QuadraticCost",
    "Trajectory",
    "continuous_dynamics",
    "euler_step",
    "load_params",
    "run_episode",
    "set_global_seeds",
]
