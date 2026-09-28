# 12 — Protocolo de la evaluación comparativa (declarado ANTES de ver resultados)

Este documento se escribió y versionó **antes** de ejecutar la evaluación, como un
pre-registro. Cualquier cambio posterior se anotará en §8 con su motivo. Implementación:
[`evaluation/protocol.py`](../src/cartpole_lab/evaluation/protocol.py) (constantes) y
[`evaluation/metrics.py`](../src/cartpole_lab/evaluation/metrics.py) (definiciones).

## 1. Qué se compara

| Método | Qué política | Corridas |
|---|---|---|
| PID, LQR, MPC, fuzzy | el controlador diseñado o sintonizado de la Fase 1 | 1 controlador (determinista) |
| Q-learning, SARSA, DQN, actor-crítico | **la política de cada una de las 10 semillas**, elegida en **validación** con la regla común (mejor punto de control; el primero ante empates) | 10 semillas |
| *Suplementarios* (tablas, no figuras principales): Double DQN, y la red **final** del actor-crítico | idem | 10 semillas |

Todos usan el mismo entorno (`CartPoleTask`, actuador continuo ±10 N), el mismo bucle
(`run_episode`) y la misma física. Las semillas de test (1000–1049) no se usaron para
elegir nada en ninguna fase.

## 2. Escenarios

| Escenario | Condiciones iniciales | Qué mide |
|---|---|---|
| **Nominal** | 50 CI de la distribución oficial (U(±0.05)⁴; semillas 1000–1049) | comportamiento "de catálogo" |
| **Impulso** | las mismas 50 CI; en t = 5 s (paso 250), una fuerza externa sobre el carro durante un paso (τ = 0.02 s), con impulso J ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0} N·s (tras la enmienda 1, §8) y signo alternado (+ en las CI pares, − en las impares) | robustez ante una perturbación externa |
| **Región de atracción** | 2 mallas de 21 × 21: (θ₀, θ̇₀) ∈ [−0.2, 0.2] rad × [−2, 2] rad/s con x = ẋ = 0; y (x₀, ẋ₀) ∈ [−2.3, 2.3] m × [−3, 3] m/s con θ = θ̇ = 0 | desde dónde se recupera cada método |

**Justificación del rango de impulsos:** partiendo del equilibrio, la búsqueda global de
rescate (`best_rescue_violation`) dio una violación mínima de 0.59 para J = 1.0 N·s y de
1.26 para J = 1.5 N·s, y se declaró 1.5 N·s como irrecuperable. **Esa afirmación resultó
falsa** (enmienda 1, §8): el barrido se amplió hasta 2.0 N·s, donde los mejores
controladores sí fallan.

## 3. Métricas (por episodio)

| Métrica | Definición |
|---|---|
| **Supervivencia** | el episodio llega a los 500 pasos |
| **Tiempo de asentamiento completo** t_s | el menor t a partir del cual, hasta el final del episodio, \|θ\| ≤ 0.5° **y** \|x\| ≤ 5 cm. "No asentado" si el episodio falla o si el estado final está fuera de la banda |
| **Tiempo de asentamiento del poste** | lo mismo, con solo \|θ\| ≤ 0.5°. Separa el control del poste del de la posición: el RL no centra el carro (Fase 2) |
| **Esfuerzo** | Σ\|F\|·τ [N·s], la fuerza del actuador, sin contar la perturbación; también Σ F²·τ y la fracción del tiempo con \|F\| = 10 N |
| **Coste común** | Σ c(s, F) con el coste normalizado de `cost.py`. **Advertencia:** es de la misma familia cuadrática que el criterio del LQR/MPC (doc 04 §4), y los favorece estructuralmente |
| **Recuperación tras el impulso** | tiempo desde el impulso hasta que \|θ\| ≤ 0.5° hasta el final (solo en los episodios que sobreviven) |
| **Pico tras el impulso** | max \|θ\| y max \|x\| después del impulso |
| **J₅₀** | el mayor impulso del barrido con supervivencia ≥ 50 % |

Las bandas (0.5°, 5 cm) son las del criterio de "estabilizado" de todo el proyecto
(`sanity.py`): una **convención**, no un estándar.

## 4. Coste de aprendizaje (solo RL)

Episodios hasta la primera evaluación greedy en validación ≥ 475 (el umbral oficial), y
pasos de entorno cuando están registrados. Se toman de los registros de la Fase 2
(`results/rl/*.json`), no se recalculan. Para los clásicos se describe cualitativamente
el coste de *diseño* (LQR: milisegundos; PID y fuzzy: ~2 min de optimización sobre el
simulador; MPC: 4.5 ms de cálculo por cada paso de control).

## 5. Estadística

- **Clásicos** (un controlador): media ± desviación típica **entre condiciones iniciales**
  (N = 50).
- **RL**: para cada semilla se promedia sobre las condiciones iniciales, y se da la
  media ± desviación típica **entre las 10 semillas** (N = 10). Así la variabilidad del
  aprendizaje queda visible y no se mezcla con la de las condiciones iniciales.
- **Región de atracción:** en los clásicos, el resultado por celda (se estabiliza /
  sobrevive / falla); en RL, la **fracción de semillas** que se estabilizan o sobreviven
  en cada celda.
- No se hacen contrastes de hipótesis formales, porque el objetivo es pedagógico. Siempre
  se indica N, y las diferencias dentro de la variabilidad entre semillas no se
  interpretan.

## 6. Qué NO se decidirá a la vista de los resultados

- Los métodos, las políticas, las condiciones iniciales, los impulsos, las mallas y las bandas quedan fijados aquí.
- **No se volverá a entrenar ni a sintonizar ningún método** en esta fase.
- Las figuras mostrarán los ocho métodos principales. Los suplementarios aparecerán en tablas.

## 7. Figuras previstas (paso 13)

1. Resumen nominal: supervivencia, asentamiento (del poste y completo) y esfuerzo por método.
2. Robustez: supervivencia frente al impulso J, por método (en RL, con la dispersión entre semillas).
3. Regiones de atracción: 2 × 8 mapas pequeños.
4. Coste de aprendizaje del RL.
5. Una tabla resumen para la diapositiva de cierre.

## 8. Registro de cambios posteriores

**Enmienda 1: rango de impulsos (antes de la evaluación completa).**
- **Qué pasó:** la prueba de humo (4 CI por método) mostró que PID, LQR y MPC **sobreviven a
  J = 1.5 N·s**, el valor que §2 declaraba irrecuperable. **El control de la medida hizo su
  trabajo: la afirmación era falsa.**
- **Causa:** `best_rescue_violation` es una búsqueda heurística, y su resultado es una cota
  *superior* de la violación mínima. Un valor > 1 era evidencia, no demostración, y aquí
  resultó demasiado pesimista.
- **Comprobación adicional:** con 10 CI, LQR y PID sobreviven 10/10 a 1.5 N·s y 0/10 a 2.0
  N·s (y 0/10 a 2.5–4.0). El límite real de los mejores controladores está entre 1.5 y
  2.0 N·s. Con 2.0 N·s, la búsqueda da 4.45: un margen amplio.
- **Cambio:** se añaden **1.75 y 2.0 N·s** al barrido. Afecta solo al rango de medida, y
  por igual a todos los métodos. Se decidió tras la prueba de humo y la comprobación con
  LQR y PID, y así se declara. Ningún otro elemento del protocolo cambia.
