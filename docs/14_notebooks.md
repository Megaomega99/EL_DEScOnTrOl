# 14 — Notebooks de la charla (Fase 4)

Dos notebooks con el **mismo orden que la charla**: entorno → control clásico → aprendizaje por
refuerzo → comparación. Difieren en qué recalculan:

| | [`01_charla_completa.ipynb`](../notebooks/01_charla_completa.ipynb) | [`02_charla_resumen.ipynb`](../notebooks/02_charla_resumen.ipynb) |
|---|---|---|
| Para qué | enseñar cómo se obtiene cada resultado | presentar los resultados sin esperas |
| Linealización, diseño del LQR, MPC, simulaciones | en vivo | en vivo (son segundos) |
| Sintonización del PID y del fuzzy (evolución diferencial, 3 ejecuciones) | en vivo (`RESINTONIZAR`) | se cargan de `results/tuning/` |
| RL: Q-learning, SARSA, DQN y actor-crítico | **una** semilla de cada uno, en vivo (`REENTRENAR`) | se cargan de `results/policies/` y `results/rl/` |
| Evaluación comparativa (10 semillas por método de RL, 1728 tareas) | se carga; `REEVALUAR = True` relanza los scripts | se carga de `results/metrics/` |
| Duración | la marcan la sintonización y los entrenamientos (ver §3) | ~15 s |

La versión corta **no entrena ni sintoniza nada**. Un test lo garantiza: sustituye las funciones de
entrenamiento y sintonización por otras que fallan, y ejecuta el notebook entero.

## 1. Decisiones de diseño

- **Por qué dos versiones.** En la charla no siempre hay diez minutos para esperar a un optimizador. La
  versión corta cuenta lo mismo con los resultados registrados. La completa muestra que cada número se
  puede recalcular con el código del repositorio.
- **En la versión completa se entrena una semilla, no diez.** Entrenar las 10 semillas de los cuatro métodos
  de RL principales suma 1.2 h de cómputo (suma de `wall_time_s` en `results/rl/*.json`; 3.0 h con
  el estudio de discretizaciones y Double DQN). Con una semilla se ve
  el proceso: la curva en vivo se dibuja **sobre** las diez curvas registradas, que dan la dispersión.
  La semilla elegida es la exportada para el navegador (doc 11), la misma que enseña la versión corta.
- **La comparación final usa siempre las políticas guardadas**, también en la versión completa. Es lo
  que mide el protocolo pre-registrado (doc 12); una semilla reentrenada en otra máquina no lo cambia.
- **El código no trivial vive en el paquete**, en [`teaching.py`](../src/cartpole_lab/teaching.py):
  simular varios controladores desde la misma CI, dibujar x/θ/F, el mapa F(θ, θ̇) con el carro centrado,
  comparar con el registro y leer resultados. Las dos versiones usan las mismas funciones, probadas en
  `tests/test_teaching.py`, y no pueden divergir.
- **Sin salidas versionadas.** Los `.ipynb` se guardan sin salidas: los diffs son legibles y las figuras
  se generan al ejecutarlos. Un test lo comprueba.
- **Los números los imprime el código.** El texto resume las justificaciones de `docs/` y, cuando cita
  un valor del registro, lo dice.

## 2. Reproducibilidad entre plataformas (comprobado, causa no aislada)

Los registros de `results/` se generaron en Windows 11 (Python 3.13, versiones de `requirements.txt`).
En esa máquina, `--regenerate-artifacts` comprueba que una semilla se reproduce exactamente
(docs 07_08, 09 y 10).

En otra plataforma no se reproduce bit a bit. En una prueba en Linux (contenedor de 4 núcleos, mismas
versiones fijadas) con la semilla exportada de cada método:

| Método | Resultado frente al registro |
|---|---|
| Q-learning (semilla 0) | curva de validación **idéntica** en las 21 primeras evaluaciones (5 250 episodios); diverge a partir del episodio 5 500 |
| DQN (semilla 1) | distinta desde la primera evaluación (5 000 pasos) |
| Actor-crítico (semilla 0) | distinta; mejor red en el paso 143 360 frente a 163 840 |

Interpretación: el Q-learning, que solo usa NumPy y Gymnasium, coincide durante miles de episodios y
luego se separa. Encaja con pequeñas diferencias de redondeo entre plataformas (p. ej. en las funciones
trigonométricas de la biblioteca matemática), que un aprendizaje por ensayo y error amplifica. **No se
ha aislado la causa**: es una hipótesis. Consecuencias:

- El registro sigue siendo la referencia. Las conclusiones de los docs se apoyan en 10 semillas, no en
  una.
- La versión completa **no exige** igualdad. Imprime si la ejecución en vivo coincide con el registro y
  dibuja la curva sobre la dispersión de las 10 semillas registradas.
- Las simulaciones con políticas **ya entrenadas** (toda la versión corta y la comparación final) solo
  dependen de la física y de la inferencia, no de un entrenamiento.

## 3. Duración

Tiempos registrados en la máquina de desarrollo (los imprime la celda de configuración de la versión
completa): sintonización del PID 112 s y del fuzzy 120 s; una semilla de Q-learning o de SARSA ~120 s,
de DQN ~177 s y de actor-crítico ~25 s.

Con todas las banderas en `True` (salvo `REEVALUAR`), la suma registrada es de unos 11 minutos (112 + 120 + 2 × 120 + 177 + 25 s), más las simulaciones.

## 4. Uso

```bash
python -m pip install -r requirements.txt && python -m pip install -e . --no-deps
jupyter lab notebooks/
```

Ejecutar de arriba abajo. En la versión completa, las banderas de la segunda celda deciden qué se
recalcula. `REEVALUAR = True` sobrescribe `results/metrics/` y las figuras `13_*`.

## 5. Verificación

```bash
python -m pytest tests/test_notebooks.py tests/test_teaching.py
```

- Los dos notebooks: formato válido, sin salidas, todas las celdas compilan y siguen el orden de la charla.
- La versión corta se ejecuta entera (≈ 15 s) y se comprueba que no entrena ni sintoniza.
- La versión completa no se ejecuta en los tests (dura minutos). Se ejecutó entera con `jupyter nbconvert
  --execute` al crearla (§3).
