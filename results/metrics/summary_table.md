| Método | Sobrevive [%] | Poste asentado [%] | Asentado completo [%] | Esfuerzo [N·s] | J₅₀ [N·s] | Región que sobrevive (malla poste / carro) | Episodios simulados para diseñarlo o aprenderlo |
|---|---|---|---|---|---|---|---|
| PID | 100 | 100 | 100 | 1.1 | 1.50 | 76% / 45% | ~1.3 M (sintonía) |
| LQR | 100 | 100 | 100 | 0.33 | 1.50 | 73% / 44% | 0 (ecuaciones) |
| MPC | 100 | 100 | 100 | 0.33 | 1.50 | 83% / 69% | 0 (ecuaciones) |
| Fuzzy | 100 | 100 | 100 | 0.87 | 1.25 | 72% / 46% | ~1.4 M (sintonía) |
| Q-learning | 93 ± 6 | 0 ± 0 | 0 ± 0 | 71 ± 10 | 1.10 ± 0.17 | 29% / 7% | ~5 800 [10/10] |
| SARSA | 78 ± 28 | 0 ± 0 | 0 ± 0 | 66 ± 15 | 0.85 ± 0.34 | 24% / 6% | ~7 700 [7/10] |
| DQN | 79 ± 42 | 6.6 ± 20.9 | 0.4 ± 1.3 | 54 ± 21 | 1.07 ± 0.59 | 49% / 30% | nunca [0/10] |
| Actor-crítico | 100 ± 0 | 71 ± 46 | 0.2 ± 0.6 | 0.87 ± 0.47 | 1.27 ± 0.08 | 37% / 18% | ~2 200 [10/10] |
