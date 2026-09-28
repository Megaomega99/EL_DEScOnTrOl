# 11 — Exportación de las políticas aprendidas para inferencia en el navegador

Pensado para quien construya el artifact web: **solo inferencia, sin reentrenar**. Todo
está en [`results/weights/`](../results/weights/), en JSON plano, sin dependencias.

**Por qué JSON y no ONNX** (la especificación admitía los dos):
- **Las redes son diminutas:** 4-64-64-5, con 4 805 parámetros, y el actor 4-64-64-1, con 4 545.
  Su inferencia (§2) son tres productos matriz-vector y un ReLU: unas diez líneas de
  JavaScript, sin motor de ejecución. ONNX obligaría a cargar onnxruntime-web, un motor de varios
  MB, para hacer lo mismo.
- **Las políticas tabulares no son redes:** son tablas con los bordes de sus cajas, y ONNX no
  las representa de forma natural. Con JSON, un solo formato sirve para todas.
- **El JSON ya está probado:** la inferencia de referencia en NumPy lee exactamente estos
  archivos, y los tests comprueban que reproduce las salidas de la red de PyTorch (valores Q y
  acción media) con un error menor de 10⁻⁵.

## 1. Archivos

| Archivo | Método | Cómo se eligió | Test (50 CI oficiales) |
|---|---|---|---|
| `tabular_q_learning.json` | Q-learning, tabla 162 × 5 | mejor en validación (semilla 0, ep. 6500) | ver doc 07_08 |
| `tabular_sarsa.json` | SARSA, tabla 162 × 5 | mejor en validación (semilla 0, ep. 5750) | ver doc 07_08 |
| `dqn.json` | DQN, red 4-64-64-5 | mejor en validación (semilla 1, paso 30 000) | 96 % |
| `double_dqn.json` | Double DQN, red 4-64-64-5 | mejor en validación (semilla 0, paso 5 000) | 100 % (casi siempre a ±10 N) |
| `actor_critic.json` | A2C, actor 4-64-64-1 | regla común: primera red con 500 en validación | 100 % (carro a ~1.2 m) |
| **`actor_critic_final.json`** | A2C, actor 4-64-64-1 | red final de la misma semilla | **100 %, carro a ~0.45 m: la recomendable** |

Ninguna red se eligió mirando el test. Todas se eligieron con las semillas de validación
2000–2009.

## 2. Formato de las redes (`"format": "mlp-json-v1"`)

```text
x = s / input_scales                      # s = [x, ẋ, θ, θ̇] del entorno, en unidades SI
para cada capa k:
    x = W_k · x + b_k                     # W_k tiene forma (salidas, entradas)
    si activation == "relu": x = max(x, 0)
DQN / Double DQN  ("force_levels"):  F = force_levels[argmax(x)]      (5 salidas)
Actor-crítico     ("output": "tanh"): F = force_limit · tanh(x[0])    (1 salida)
```

Pesos en float32. `input_scales` = [2.4, 1.0, 0.2094, 1.5]. En `metadata` constan la semilla,
el punto de control, el retorno en validación y la configuración de entrenamiento.

## 3. Formato de las tablas (tabular)

```text
caja de cada variable v:  número de bordes de edges[v] que son ≤ valor (un valor igual a un borde va arriba)
índice = ravel_multi_index(cajas, forma = (len(edges[v]) + 1 for v))     # orden C (fila mayor)
F = force_levels[argmax(q_table[índice])]; empates -> la fuerza más suave (0, −5, +5, −10, +10)
```

## 4. Garantía de fidelidad

- La inferencia de referencia es `network_forward_numpy` (NumPy, float32). Es **la misma
  que se usó para evaluar** durante el entrenamiento.
- Un test comprueba que coincide con la red de PyTorch (tolerancia 10⁻⁵):
  `test_exported_json_reproduces_the_torch_network` y
  `test_exported_actor_reproduces_the_torch_mean_action`.
- Una implementación en JavaScript debe reproducir las acciones de la referencia de NumPy.
  Conviene validarla con estados de prueba antes de publicarla.
