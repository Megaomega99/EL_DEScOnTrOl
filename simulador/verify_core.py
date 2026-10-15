"""Verifica que el núcleo JavaScript del simulador (simulador/core.js) reproduce el código Python.

Por qué existe: el simulador del celular debe mostrar los MISMOS controladores que se
compararon en la charla. Este script calcula referencias con cartpole_lab (dinámica,
PID, fuzzy, LQR, MPC con CVXPY, políticas tabulares y redes) y las contrasta con lo que
devuelve core.js ejecutado en Node. Uso:  python simulador/verify_core.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from cartpole_lab.controllers.fuzzy import CascadeFuzzy, load_tuned_fuzzy_gains
from cartpole_lab.controllers.lqr import design_default_lqr
from cartpole_lab.controllers.mpc import LinearMPC, MPCConfig
from cartpole_lab.controllers.pid import CascadePID, load_tuned_gains
from cartpole_lab.dynamics import euler_step
from cartpole_lab.params import load_params
from cartpole_lab.rl.discretization import BOXES_1983
from cartpole_lab.rl.tabular import GreedyTabularPolicy

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

NODE_SIDE = r"""
const Core = require(process.argv[2]);
const ref = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = {};
out.step = ref.step.map(([s, f]) => Core.step(s, f));
const pid = Core.makePID(ref.pid_gains);
out.pid = ref.traj.map((s) => pid.act(s).force);
const fz = Core.makeFuzzy(ref.fuzzy_gains, {e: [-1, -0.5, 0, 0.5, 1], d: [-1, -0.5, 0, 0.5, 1]});
out.fuzzy = ref.traj.map((s) => fz.act(s).force);
const w = Core.brysonWeights(1.2, Math.acos(0.99), 10);
out.K = Array.from(Core.dlqr(w.q, w.r).K);
out.mpc = ref.mpc_cases.map((c) => {
  const m = Core.makeMPC({horizon: c.N, q: w.q, r: w.r, forceLimit: 10, xLimit: c.xlim, thetaLimit: c.thlim});
  const sol = m.solve(c.s0);
  return {u0: sol.force, iters: sol.iters, converged: sol.converged, forces: sol.forces};
});
const tab = Core.makeTabularPolicy(ref.q_table);
out.tab = ref.traj.map((s) => tab.act(s).force);
out.box = ref.traj.map((s) => Core.boxIndex(s));
out.dqn = ref.traj.map((s) => Core.mlpForward(ref.dqn, s));
out.ac = ref.traj.map((s) => Core.makeNetPolicy(ref.ac).act(s).force);
process.stdout.write(JSON.stringify(out));
"""


def mlp(net: dict, s: np.ndarray) -> np.ndarray:
    x = s / np.asarray(net["input_scales"])
    for i, layer in enumerate(net["layers"]):
        x = np.asarray(layer["W"]) @ x + np.asarray(layer["b"])
        if i < len(net["layers"]) - 1:
            x = np.maximum(x, 0.0)
    return x


def main() -> None:
    params = load_params()
    rng = np.random.default_rng(0)
    steps = [(rng.uniform([-2, -2, -0.2, -2], [2, 2, 0.2, 2]), float(rng.uniform(-12, 12))) for _ in range(200)]
    traj = [rng.uniform([-2.3, -3, -0.2, -2], [2.3, 3, 0.2, 2]) for _ in range(300)]
    traj += [rng.uniform(-0.05, 0.05, 4) for _ in range(100)]

    pid_gains, fuzzy_gains = load_tuned_gains(), load_tuned_fuzzy_gains()
    pid = CascadePID(pid_gains, params)
    fuzzy = CascadeFuzzy(fuzzy_gains, params)
    q_table = np.asarray(json.loads((ROOT / "results/weights/tabular_q_learning.json").read_text())["q_table"])
    tab = GreedyTabularPolicy(q_table, BOXES_1983)
    dqn = json.loads((ROOT / "results/weights/dqn.json").read_text())
    ac = json.loads((ROOT / "results/weights/actor_critic.json").read_text())

    mpc_cases = []
    for N, xlim, thlim, s0 in [
        (25, 2.4, None, [0.1, 0.0, 0.05, 0.0]),
        (25, 2.4, None, [0.0, 0.0, 0.15, 0.5]),
        (25, 2.4, None, [-1.5, 1.0, -0.1, 1.0]),
        (25, 0.5, None, [0.0, 1.5, 0.05, 0.0]),
        (10, 2.4, 0.05, [0.3, 0.0, 0.04, 0.4]),
        (50, 1.0, None, [0.8, 0.5, -0.12, -0.3]),
    ]:
        config = MPCConfig(horizon=N, x_constraint_m=xlim, theta_constraint_rad=thlim)
        mpc = LinearMPC(params, config)
        u0 = mpc(np.asarray(s0))
        mpc_cases.append({"N": N, "xlim": xlim, "thlim": config.theta_constraint(params), "s0": s0,
                          "u0_ref": u0, "forces_ref": mpc.last_plan["forces"].tolist()})

    ref = {
        "step": [[s.tolist(), f] for s, f in steps],
        "traj": [s.tolist() for s in traj],
        "pid_gains": pid_gains.to_dict(),
        "fuzzy_gains": fuzzy_gains.to_dict(),
        "mpc_cases": mpc_cases,
        "q_table": q_table.tolist(),
        "dqn": dqn,
        "ac": ac,
    }
    script = HERE / "_node_side.js"
    script.write_text(NODE_SIDE, encoding="utf-8")
    try:
        result = subprocess.run(["node", str(script), str(HERE / "core.js")], input=json.dumps(ref),
                                capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stderr)
    finally:
        script.unlink()
    js = json.loads(result.stdout)

    report = {}
    report["paso de Euler (máx |Δ|)"] = max(np.max(np.abs(np.asarray(j) - euler_step(s, f, params)))
                                           for j, (s, f) in zip(js["step"], steps))
    pid.reset()
    report["PID (máx |ΔF| N)"] = max(abs(j - pid(np.asarray(s))) for j, s in zip(js["pid"], traj))
    report["fuzzy (máx |ΔF| N)"] = max(abs(j - fuzzy(np.asarray(s))) for j, s in zip(js["fuzzy"], traj))
    K_ref = design_default_lqr(params).K.ravel()
    report["LQR (máx |ΔK| relativo)"] = float(np.max(np.abs(np.asarray(js["K"]) - K_ref) / np.abs(K_ref)))
    report["MPC F0 (máx |ΔF| N)"] = max(abs(j["u0"] - c["u0_ref"]) for j, c in zip(js["mpc"], mpc_cases))
    report["MPC plan (máx |ΔF| N)"] = max(float(np.max(np.abs(np.asarray(j["forces"]) - c["forces_ref"])))
                                         for j, c in zip(js["mpc"], mpc_cases))
    report["MPC iteraciones ADMM"] = [j["iters"] for j in js["mpc"]]
    report["MPC convergió"] = all(j["converged"] for j in js["mpc"])
    report["cajas distintas"] = sum(j != BOXES_1983.index(s) for j, s in zip(js["box"], traj))
    report["tabular acciones distintas"] = sum(j != tab(np.asarray(s)) for j, s in zip(js["tab"], traj))
    report["DQN (máx |ΔQ|)"] = max(float(np.max(np.abs(np.asarray(j) - mlp(dqn, s)))) for j, s in zip(js["dqn"], traj))
    report["actor-crítico (máx |ΔF| N)"] = max(abs(j - ac["force_limit"] * np.tanh(mlp(ac, s)[0]))
                                              for j, s in zip(js["ac"], traj))
    for key, value in report.items():
        print(f"{key:32s} {value}")


if __name__ == "__main__":
    main()
