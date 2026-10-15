"""Construye el simulador interactivo (un único HTML autocontenido) a partir de results/.

Por qué existe: el simulador de la charla debe usar EXACTAMENTE los parámetros sintonizados
y las políticas entrenadas del repositorio, no copias a mano. Este script los lee de
results/ (ganancias del PID y del fuzzy, configuración del MPC, tablas Q, redes DQN y
actor-crítico, curvas de aprendizaje), los incrusta junto con core.js en app.html y
escribe una sola página que funciona sin servidor, también en el celular.

Uso:  python simulador/build.py [salida.html]      (por defecto simulador/dist/simulador.html)
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"


def load(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


def rounded(obj, digits: int = 6):
    """Redondea a `digits` cifras significativas: reduce a la mitad el peso de las redes.

    El error relativo (5·10⁻⁷) es varios órdenes menor que las diferencias entre acciones
    que deciden la política; verify_core.py comprueba las redes sin redondear.
    """
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, list):
        return [rounded(v, digits) for v in obj]
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    return obj


def policy(net: dict) -> dict:
    """Solo lo necesario para la inferencia (sin metadatos de entrenamiento)."""
    keys = ("input_scales", "force_levels", "force_limit", "layers", "q_table", "edges")
    return rounded({k: net[k] for k in keys if k in net})


def logo_mask() -> str:
    """Máscara del logo (trazos del original): la página la colorea con el color de texto del tema."""
    rgba = np.asarray(Image.open(ROOT / "assets" / "logo-topbar.png").convert("RGBA")).astype(float)
    alpha = np.clip((rgba[..., 2] - 150.0) / 80.0, 0.0, 1.0) * rgba[..., 3] / 255.0  # quita el halo naranja
    out = np.zeros(rgba.shape, dtype=np.uint8)
    out[..., 3] = (alpha * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(out).resize((320, 70), Image.LANCZOS).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_data() -> dict:
    pid, fuzzy, mpc = load("tuning/pid.json"), load("tuning/fuzzy.json"), load("tuning/mpc.json")
    tabular, dqn, ac = load("rl/tabular.json"), load("rl/dqn.json"), load("rl/actor_critic.json")
    q_w, s_w = load("weights/tabular_q_learning.json"), load("weights/tabular_sarsa.json")
    dqn_best, ac_best = load("weights/dqn.json"), load("weights/actor_critic.json")
    dqn_final = load(f"policies/dqn/seed_{dqn_best['metadata']['seed']:02d}_final.json")
    ac_final = load(f"policies/actor_critic/seed_{ac_best['metadata']['seed']:02d}_final.json")

    def tab_runs(method: str) -> list[dict]:
        return [{"seed": r["seed"], "x": r["eval_episodes"], "y": rounded(r["eval_mean"], 4)}
                for r in tabular["runs"] if r["method"] == method and r["discretization"] == "boxes_1983"]

    def run_of(runs: list[dict], seed: int, **match) -> dict:
        return next(r for r in runs if r["seed"] == seed and all(r.get(k) == v for k, v in match.items()))

    def survival(block: dict) -> float:
        return round(block["survival_rate"]["mean"], 3)

    dqn_run = run_of(dqn["runs"], dqn_best["metadata"]["seed"], variant="dqn")
    ac_run = run_of(ac["runs"], ac_best["metadata"]["seed"])
    q_run = run_of(tabular["runs"], q_w["seed"], method="q_learning", discretization="boxes_1983")
    s_run = run_of(tabular["runs"], s_w["seed"], method="sarsa", discretization="boxes_1983")
    agents = {
        "q": {"name": "Q-learning", "kind": "tabla", "unit": "episodios", "policy": policy(q_w),
              "curve": {"x": q_run["eval_episodes"], "y": rounded(q_run["eval_mean"], 4)},
              "checkpoint": q_w["checkpoint_episode"], "seed": q_w["seed"],
              "survival": survival(tabular["methods"]["q_learning"]["test_best_checkpoint"])},
        "sarsa": {"name": "SARSA", "kind": "tabla", "unit": "episodios", "policy": policy(s_w),
                  "curve": {"x": s_run["eval_episodes"], "y": rounded(s_run["eval_mean"], 4)},
                  "checkpoint": s_w["checkpoint_episode"], "seed": s_w["seed"],
                  "survival": survival(tabular["methods"]["sarsa"]["test_best_checkpoint"])},
        "dqn": {"name": "DQN · mejor red", "kind": "red", "unit": "pasos", "policy": policy(dqn_best),
                "curve": {"x": dqn_run["eval_steps"], "y": rounded(dqn_run["eval_mean"], 4)},
                "checkpoint": dqn_best["metadata"]["checkpoint_step"], "seed": dqn_best["metadata"]["seed"],
                "survival": survival(dqn["summary"]["dqn"]["test_best_checkpoint"])},
        "dqnFinal": {"name": "DQN · red final", "kind": "red", "unit": "pasos", "policy": policy(dqn_final),
                     "curve": {"x": dqn_run["eval_steps"], "y": rounded(dqn_run["eval_mean"], 4)},
                     "checkpoint": dqn["configs"]["dqn"]["total_steps"], "seed": dqn_best["metadata"]["seed"],
                     "survival": survival(dqn["summary"]["dqn"]["test_final_network"])},
        "ac": {"name": "Actor-crítico · mejor red", "kind": "red", "unit": "pasos", "policy": policy(ac_best),
               "curve": {"x": ac_run["eval_steps"], "y": rounded(ac_run["eval_mean"], 4)},
               "checkpoint": ac_best["metadata"]["checkpoint_step"], "seed": ac_best["metadata"]["seed"],
               "survival": survival(ac["summary"]["test_best_checkpoint"])},
        "acFinal": {"name": "Actor-crítico · red final", "kind": "red", "unit": "pasos", "policy": policy(ac_final),
                    "curve": {"x": ac_run["eval_steps"], "y": rounded(ac_run["eval_mean"], 4)},
                    "checkpoint": ac["config"]["total_steps"], "seed": ac_best["metadata"]["seed"],
                    "survival": survival(ac["summary"]["test_final_network"])},
    }
    return {
        "pid": pid["gains"],
        "fuzzy": fuzzy["gains"],
        "mpc": {"horizon": mpc["config"]["horizon"], "thetaLimit": mpc["theta_constraint_rad"],
                "xLimit": mpc["x_constraint_m"], "slackWeight": mpc["config"]["slack_weight"]},
        "tabularLab": {"q_learning": tab_runs("q_learning"), "sarsa": tab_runs("sarsa"),
                       "episodes": tabular["n_episodes"], "evalEvery": tabular["configs"]["q_learning"]["eval_every"]},
        "agents": agents,
        "logo": logo_mask(),
    }


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "dist" / "simulador.html"
    page = (HERE / "app.html").read_text(encoding="utf-8")
    core = (HERE / "core.js").read_text(encoding="utf-8")
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    for marker in ("/*__CORE__*/", "/*__DATA__*/"):
        if page.count(marker) != 1:
            raise SystemExit(f"app.html debe contener {marker} exactamente una vez")
    page = page.replace("/*__CORE__*/", core).replace("/*__DATA__*/", f"const DATA = {data};")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
