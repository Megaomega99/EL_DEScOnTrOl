"""Notebooks docentes (Fase 4): estructura, limpieza y ejecución completa de la versión corta.

La versión completa entrena y sintoniza (~10 min): aquí solo se comprueba que su código compila.
La corta se ejecuta entera, y además se verifica que NO entrena ni sintoniza nada.
"""

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from cartpole_lab.paths import REPO_ROOT

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
FULL = NOTEBOOKS_DIR / "01_charla_completa.ipynb"
SHORT = NOTEBOOKS_DIR / "02_charla_resumen.ipynb"
TALK_ORDER = ("## 1. El entorno", "## 2. Control clásico", "## 3. Aprendizaje por refuerzo", "## 4. Comparación")


def _cells(path, cell_type):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == cell_type]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


@pytest.mark.parametrize("path", [FULL, SHORT], ids=["completo", "corto"])
def test_notebook_is_clean_and_every_code_cell_compiles(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(_cells(path, "code")):
        # Sin salidas versionadas: se generan al ejecutarlo (y no ensucian los diffs).
        assert cell["outputs"] == [] and cell["execution_count"] is None, f"celda {index} con salidas"
        compile(_source(cell), f"{path.name}[{index}]", "exec")


@pytest.mark.parametrize("path", [FULL, SHORT], ids=["completo", "corto"])
def test_notebook_follows_the_talk_order(path):
    headings = [line for cell in _cells(path, "markdown") for line in _source(cell).splitlines()
                if line.startswith("## ")]
    positions = [next(i for i, h in enumerate(headings) if h.startswith(section)) for section in TALK_ORDER]
    assert positions == sorted(positions)


def test_full_notebook_declares_what_it_recomputes():
    setup = "\n".join(_source(c) for c in _cells(FULL, "code"))
    for flag in ("RESINTONIZAR", "REENTRENAR", "REEVALUAR"):
        assert re.search(rf"^{flag} = (True|False)", setup, flags=re.MULTILINE), flag


def test_short_notebook_runs_end_to_end_without_training_or_tuning(monkeypatch):
    pytest.importorskip("IPython")
    import cartpole_lab.controllers.fuzzy_tuning as fuzzy_tuning
    import cartpole_lab.controllers.pid_tuning as pid_tuning
    import cartpole_lab.rl.actor_critic as actor_critic
    import cartpole_lab.rl.dqn as dqn
    import cartpole_lab.rl.tabular as tabular

    def forbidden(*args, **kwargs):
        raise AssertionError("La versión corta no debe entrenar ni sintonizar")

    for module, name in ((tabular, "train_tabular"), (dqn, "train_dqn"), (actor_critic, "train_a2c"),
                         (pid_tuning, "tune_pid"), (fuzzy_tuning, "tune_fuzzy")):
        monkeypatch.setattr(module, name, forbidden)
    # En Jupyter, plt.show() muestra y cierra cada figura; aquí solo se cierran (Agg no muestra nada).
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    namespace: dict = {}
    try:
        for index, cell in enumerate(_cells(SHORT, "code")):
            exec(compile(_source(cell), f"{SHORT.name}[{index}]", "exec"), namespace)
    finally:
        plt.close("all")
    assert set(namespace["tray_impulso"]) == {"PID", "LQR", "MPC", "Fuzzy", "Q-learning", "SARSA", "DQN", "Actor-crítico"}
