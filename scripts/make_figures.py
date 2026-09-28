"""Paso 13: figuras comparativas finales para las diapositivas, a partir de results/metrics/comparison.json.

Uso:  python scripts/make_figures.py      (segundos; no simula nada)

Figuras (results/figures/) y tabla (results/metrics/):
    13_resumen_nominal.png        supervivencia, asentamiento y esfuerzo de los 8 métodos
    13_robustez_impulso.png       supervivencia frente al impulso lateral J (clásicos | RL)
    13_regiones_atraccion.png     desde qué estados iniciales sobrevive (y se estabiliza) cada método
    13_coste_aprendizaje.png      episodios de experiencia que necesita cada método de RL
    13_tabla_resumen.png          tabla para la diapositiva de cierre
    summary_table.md              la misma tabla en Markdown (la cita docs/13)
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cartpole_lab.evaluation.protocol import CART_GRID_AXES, IMPULSES_NS, MAIN_METHODS, POLE_GRID_AXES, RL_SEEDS
from cartpole_lab.paths import FIGURES_DIR, RESULTS_DIR, TUNING_DIR
from cartpole_lab.plotting import METHOD_COLORS, MUTED, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

SOURCE = RESULTS_DIR / "metrics" / "comparison.json"
TABLE_MD = RESULTS_DIR / "metrics" / "summary_table.md"
# Secuencial de un solo tono (azul de la paleta de referencia): fracción de 0 a 1.
SEQUENTIAL = LinearSegmentedColormap.from_list("fraccion", ["#f4f3ef", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
CLASSICAL = ("PID", "LQR", "MPC", "Fuzzy")
RL_MAIN = ("Q-learning", "SARSA", "DQN", "Actor-crítico")
# Dominio válido de cada tipo de métrica: las barras media ± desv. típica se recortan a él.
DOMAINS = {"rate": (0.0, 1.0), "time": (0.0, 10.0), "log": (1e-3, np.inf)}


def scalar(entry) -> float:
    """Valor escalar de una entrada del JSON: número, o la media de un resumen {mean, std, n}."""
    if isinstance(entry, dict):
        return np.nan if entry["mean"] is None else float(entry["mean"])
    return float(entry)


def value(block: dict, key: str) -> tuple[float, float]:
    """(valor, desv. típica). Clásicos: estadística entre CI (o tasa, sin barra); RL: entre semillas."""
    entry = block[key]
    if isinstance(entry, dict):
        return scalar(entry), float(entry["std"] or 0.0)
    return float(entry), 0.0


def seed_values(method: dict, key: str) -> list[float]:
    """El valor de cada semilla (solo RL): la dispersión real, no solo su resumen."""
    per_seed = method.get("per_seed_nominal") or {}
    return [scalar(per_seed[str(s)][key]) for s in RL_SEEDS if str(s) in per_seed]


def clipped_err(v: float, e: float, kind: str) -> np.ndarray:
    lo, hi = DOMAINS[kind]
    return np.array([[min(e, max(v - lo, 0.0))], [min(e, max(hi - v, 0.0))]])


def _nominal_panel(ax, methods: dict, key: str, kind: str, y: np.ndarray) -> None:
    for yi, name in zip(y, MAIN_METHODS):
        m = methods[name]
        v, e = value(m["nominal"], key)
        color = METHOD_COLORS[name]
        if np.isnan(v):
            ax.text(0.02, yi, "nunca", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=TEXT_SECONDARY)
            continue
        if m["is_rl"]:
            seeds = seed_values(m, key)
            offsets = np.linspace(-0.22, 0.22, len(seeds))
            ax.scatter(seeds, yi + offsets, s=14, color=color, alpha=0.35, linewidths=0, zorder=2)
            n_seeds = m["nominal"][key].get("n", len(seeds))
            if n_seeds < len(RL_SEEDS):  # p. ej. un tiempo que solo existe en las semillas que se asientan
                ax.text(0.98, yi + 0.3, f"solo {n_seeds}/{len(RL_SEEDS)} semillas", transform=ax.get_yaxis_transform(),
                        ha="right", fontsize=8, color=TEXT_SECONDARY)
        ax.errorbar(v, yi, xerr=clipped_err(v, e, kind), fmt="D" if m["is_rl"] else "o", color=color, markersize=8,
                    capsize=4, elinewidth=1.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.axhline(3.5, color=MUTED, linewidth=0.8, linestyle=":")
    if kind == "rate":
        ax.set_xlim(-0.03, 1.05)
    elif kind == "log":
        ax.set_xscale("log")


def plot_nominal(methods: dict) -> None:
    apply_style()
    panels = [("survival_rate", "Sobrevive 500 pasos", "rate"),
              ("pole_settled_rate", "Poste asentado\n|θ| ≤ 0.5° durante ≥ 1 s", "rate"),
              ("settled_rate", "Asentado completo\n… y además |x| ≤ 5 cm", "rate"),
              ("pole_settling_time_s", "Tiempo de asentamiento\ndel poste [s]", "time"),
              ("effort_abs_Ns", "Esfuerzo Σ|F|·τ [N·s]\n(escala log)", "log")]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 5.2), sharey=True)
    y = np.arange(len(MAIN_METHODS))[::-1].astype(float)
    for ax, (key, title, kind) in zip(axes, panels):
        _nominal_panel(ax, methods, key, kind, y)
        ax.set_title(title, loc="left", fontsize=10.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(MAIN_METHODS)
    fig.suptitle("Escenario nominal: 50 CI de la distribución oficial.  ● clásicos (barra: entre CI)  ·  "
                 "◆ RL: media ± desv. típica entre 10 semillas; puntos claros = cada semilla",
                 fontsize=11.5, color=TEXT_SECONDARY)
    fig.text(0.01, 0.01, "LQR y MPC coinciden aquí: con CI pequeñas ninguna restricción del MPC se activa y, con coste "
             "terminal P, su solución es exactamente la del LQR (doc 05).", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES_DIR / "13_resumen_nominal.png", dpi=150)
    plt.close(fig)


def _impulse_curve(method: dict) -> tuple[np.ndarray, np.ndarray]:
    stats = [value(method["impulse"][str(j)], "survival_rate") for j in IMPULSES_NS]
    return np.array([s[0] for s in stats]), np.array([s[1] for s in stats])


def _classical_impulse_panel(ax, methods: dict, J: np.ndarray) -> None:
    # PID, LQR y MPC dan curvas IDÉNTICAS: se dibujan de más gruesa a más fina para que se vean las tres.
    for name, width, marker in (("PID", 8.0, 11), ("LQR", 4.5, 8), ("MPC", 1.8, 5), ("Fuzzy", 2.0, 6)):
        mean, _ = _impulse_curve(methods[name])
        ax.plot(J, mean, "-", marker="o", color=METHOD_COLORS[name], label=name, linewidth=width, markersize=marker,
                alpha=0.55 if name == "PID" else 1.0, solid_capstyle="round")
    ax.text(0.25, 1.07, "PID, LQR y MPC: curvas idénticas (dibujadas de más gruesa a más fina)", fontsize=9,
            color=TEXT_SECONDARY)
    ax.set_title("Control clásico (1 controlador, 50 CI por punto)", loc="left", fontsize=11)


def _rl_impulse_panel(ax, methods: dict, J: np.ndarray) -> None:
    reference, _ = _impulse_curve(methods["LQR"])
    ax.plot(J, reference, ":", color=MUTED, linewidth=2, label="LQR (referencia)")
    for name in RL_MAIN:
        mean, err = _impulse_curve(methods[name])
        ax.plot(J, mean, "--", marker="D", color=METHOD_COLORS[name], label=name, markersize=6)
        ax.fill_between(J, np.clip(mean - err, 0, 1), np.clip(mean + err, 0, 1), color=METHOD_COLORS[name],
                        alpha=0.10, linewidth=0)
    ax.annotate("J ≥ 1.75: solo sobrevive\nquien OSCILA, si el impulso\nllega a mitad de un vaivén\n"
                "y lo compensa (doc 13 §3)", xy=(1.875, 0.14), xytext=(2.02, 0.80), ha="right", fontsize=8.5,
                color=TEXT_SECONDARY, arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set_title("Aprendizaje por refuerzo (media ± 1 desv. típica entre 10 semillas)", loc="left", fontsize=11)


def plot_impulse(methods: dict) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    J = np.array(IMPULSES_NS)
    _classical_impulse_panel(axes[0], methods, J)
    _rl_impulse_panel(axes[1], methods, J)
    for ax in axes:
        ax.axvspan(1.5, 1.75, color=MUTED, alpha=0.15, linewidth=0)
        ax.text(1.625, 1.13, "límite de\nPID/LQR/MPC", ha="center", va="top", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlabel("impulso lateral sobre el carro en t = 5 s, J [N·s]")
        ax.set_ylim(-0.02, 1.15)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("fracción de episodios que sobreviven")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_robustez_impulso.png", dpi=150)
    plt.close(fig)


def _cell_extent(xs: np.ndarray, ys: np.ndarray, yscale: float) -> tuple[float, float, float, float]:
    """Bordes de las celdas (no sus centros): cada píxel queda centrado en su punto de la malla."""
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    return xs[0] - dx / 2, xs[-1] + dx / 2, (ys[0] - dy / 2) * yscale, (ys[-1] + dy / 2) * yscale


def plot_regions(methods: dict) -> None:
    apply_style()
    fig, axes = plt.subplots(2, len(MAIN_METHODS), figsize=(18, 6.6), sharex="row", sharey="row",
                             gridspec_kw={"hspace": 0.62, "wspace": 0.12})
    grids = [("pole_grid", POLE_GRID_AXES, "θ̇₀ [rad/s]", "θ₀ [°]", np.degrees(1.0)),
             ("cart_grid", CART_GRID_AXES, "ẋ₀ [m/s]", "x₀ [m]", 1.0)]
    for row, (grid, (ys, xs), xlabel, ylabel, yscale) in enumerate(grids):
        for col, name in enumerate(MAIN_METHODS):
            ax = axes[row, col]
            g = methods[name]["grids"][grid]
            mesh = ax.imshow(np.array(g["survived_fraction_per_cell"]), origin="lower", aspect="auto", cmap=SEQUENTIAL,
                             vmin=0, vmax=1, extent=_cell_extent(xs, ys, yscale), interpolation="nearest")
            ax.set_title(f"{name}\nsobrev. {g['survived_area']['mean']:.0%} · estab. {g['stabilized_area']['mean']:.0%}",
                         fontsize=9, color=TEXT_PRIMARY)
            ax.grid(False)
            ax.set_xlabel(xlabel, fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
    bar = fig.colorbar(mesh, ax=axes, fraction=0.015, pad=0.01)
    bar.set_label("fracción que SOBREVIVE 500 pasos\n(RL: fracción de las 10 semillas)", fontsize=9)
    fig.suptitle("¿Desde dónde se recupera cada método?  Arriba: poste inclinado, carro centrado  ·  "
                 "abajo: carro desplazado, poste vertical\nestab. = además queda en |θ| ≤ 0.5° y |x| ≤ 5 cm durante "
                 "el último segundo: en los clásicos casi coincide con sobrevivir; el RL sobrevive pero no centra el carro",
                 fontsize=11.5, color=TEXT_SECONDARY, y=1.04)
    fig.savefig(FIGURES_DIR / "13_regiones_atraccion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def tuning_episodes(name: str) -> int:
    """Episodios simulados que consumió la sintonización por evolución diferencial (PID y fuzzy, docs 03 y 06).

    SciPy cuenta en `nfev` llamadas VECTORIZADAS: cada una evalúa la población entera (popsize × nº de
    parámetros candidatos), y cada candidato se simula desde `n_initial_states` CI (polish=False: no hay más).
    """
    tuning = json.loads((TUNING_DIR / f"{name}.json").read_text(encoding="utf-8"))["tuning"]
    config = tuning["config"]
    population = config["popsize"] * len(config["bounds"])
    return sum(run["nfev"] for run in tuning["runs"]) * population * config["n_initial_states"]


TUNED_CLASSICAL = {"PID": "pid", "Fuzzy": "fuzzy"}  # los que se sintonizan sobre el simulador (caja negra)

LEARNING_ROWS = [("Q-learning", "episodes_to_first_greedy_475", "primera política ≥ 475"),
                 ("Q-learning", "episodes_to_moving_average_475", "media móvil ≥ 475"),
                 ("SARSA", "episodes_to_first_greedy_475", "primera política ≥ 475"),
                 ("SARSA", "episodes_to_moving_average_475", "media móvil ≥ 475"),
                 ("DQN", "episodes_to_first_greedy_475", "primera política ≥ 475"),
                 ("DQN", "episodes_to_moving_average_475", "media móvil ≥ 475"),
                 ("Actor-crítico", "episodes_to_moving_average_475", "media móvil ≥ 475")]


def _learning_row(ax, yi: int, name: str, key: str, label: str, costs: dict) -> str:
    values = [v for v in costs[name].get(key, []) if v is not None]
    n_total = len(costs[name].get(key, []))
    if values:
        ax.scatter(values, np.full(len(values), yi), color=METHOD_COLORS[name], alpha=0.6, s=28,
                   marker="o" if "primera" in label else "D")
        ax.plot(np.median(values), yi, "|", color=TEXT_PRIMARY, markersize=16, markeredgewidth=2)
    else:
        ax.text(120, yi, f"nunca (0 de {n_total} semillas)", va="center", fontsize=9, color=TEXT_SECONDARY)
    return f"{name}: {label}  [{len(values)}/{n_total} semillas]"


def plot_learning_costs(costs: dict) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    labels = [_learning_row(ax, yi, *row, costs) for yi, row in enumerate(LEARNING_ROWS[::-1])]
    for offset, (name, file) in enumerate(TUNED_CLASSICAL.items()):
        yi = len(LEARNING_ROWS) + offset
        n = tuning_episodes(file)
        ax.scatter([n], [yi], color=METHOD_COLORS[name], s=60, marker="s")
        ax.text(n / 1.5, yi, f"{n / 1e6:.1f} millones", ha="right", va="center", fontsize=9, color=TEXT_SECONDARY)
        labels.append(f"{name}: sintonía por evolución diferencial (total)")
    ax.axhline(len(LEARNING_ROWS) - 0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(100, 4e6)
    ax.set_xlabel("episodios simulados (escala log;  | = mediana entre semillas)")
    ax.set_title("¿Cuánta experiencia simulada necesita cada método?\n"
                 "LQR y MPC: ninguna (se diseñan con las ecuaciones del modelo).\n"
                 "PID y fuzzy: sintonía de caja negra sobre el simulador, con más episodios que cualquier RL.",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_coste_aprendizaje.png", dpi=150)
    plt.close(fig)


def _rl_fmt(m: dict, key: str, pattern: str | None = None, pct: bool = False) -> str:
    """Clásicos: la tasa o la media entre CI; RL: media ± desv. típica entre semillas."""
    v, e = value(m["nominal"], key)
    if np.isnan(v):
        return "—"
    scale = 100.0 if pct else 1.0
    if pattern is None:  # porcentajes: un decimal solo si el valor es pequeño pero no nulo (0.4 %, no "0 %")
        pattern = "{:.1f}" if 0 < v * scale < 10 else "{:.0f}"
    text = pattern.format(v * scale)
    return f"{text} ± {pattern.format(e * scale)}" if m["is_rl"] else text


def _experience(name: str, costs: dict) -> str:
    """RL: episodios hasta que la media móvil de 100 episodios llega a 475 (y se mantiene), mismo criterio para
    todos. PID/fuzzy: total de su sintonización. LQR/MPC: ninguno (solo las ecuaciones del modelo)."""
    if name in TUNED_CLASSICAL:
        return f"~{tuning_episodes(TUNED_CLASSICAL[name]) / 1e6:.1f} M (sintonía)"
    if name in CLASSICAL:
        return "0 (ecuaciones)"
    runs = costs[name].get("episodes_to_moving_average_475", [])
    values = [v for v in runs if v is not None]
    if not values:
        return f"nunca [0/{len(runs)}]"
    # Redondeo a la centena: con 10 semillas y dispersión de miles, más cifras serían falsa precisión.
    return f"~{round(float(np.median(values)), -2):,.0f} [{len(values)}/{len(runs)}]".replace(",", " ")


def summary_rows(methods: dict, costs: dict) -> list[list[str]]:
    rows = []
    for name in MAIN_METHODS:
        m = methods[name]
        grids = m["grids"]
        rows.append([name, _rl_fmt(m, "survival_rate", pct=True), _rl_fmt(m, "pole_settled_rate", pct=True),
                     _rl_fmt(m, "settled_rate", pct=True), _rl_fmt(m, "effort_abs_Ns", "{:.2g}"),
                     f"{m['j50']['mean']:.2f}" + (f" ± {m['j50']['std']:.2f}" if m["is_rl"] else ""),
                     f"{grids['pole_grid']['survived_area']['mean']:.0%} / {grids['cart_grid']['survived_area']['mean']:.0%}",
                     _experience(name, costs)])
    return rows


SUMMARY_HEADER = ["Método", "Sobrevive\n[%]", "Poste asentado\n[%]", "Asentado\ncompleto [%]", "Esfuerzo\n[N·s]",
                  "J₅₀\n[N·s]", "Región que sobrevive\n(malla poste / carro)", "Episodios simulados\npara diseñarlo o aprenderlo"]
SUMMARY_COL_WIDTHS = [0.10, 0.09, 0.11, 0.11, 0.10, 0.11, 0.17, 0.21]


def plot_summary_table(methods: dict, costs: dict) -> None:
    apply_style()
    rows = summary_rows(methods, costs)
    fig, ax = plt.subplots(figsize=(17, 4.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=SUMMARY_HEADER, colWidths=SUMMARY_COL_WIDTHS, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#e3e2dc")
        if r == 0:
            cell.set_text_props(color=TEXT_PRIMARY, weight="bold")
            cell.set_facecolor("#f1f0ec")
            cell.set_height(cell.get_height() * 1.7)
        elif c == 0:
            cell.set_text_props(color=METHOD_COLORS[rows[r - 1][0]], weight="bold")
    ax.set_title("Resumen: 50 CI nominales · RL: media ± desv. típica entre 10 semillas · J₅₀: mayor impulso con "
                 "≥ 50 % de supervivencia\nEpisodios: RL, hasta que la media móvil llega a 475 y se mantiene "
                 "([n/10]: semillas que lo consiguen); PID y fuzzy, total de su sintonía", loc="left", fontsize=11,
                 color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_tabla_resumen.png", dpi=150)
    plt.close(fig)
    header = [h.replace("\n", " ") for h in SUMMARY_HEADER]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    methods = record["methods"]
    missing = [m for m in MAIN_METHODS if m not in methods]
    if missing:
        raise KeyError(f"Faltan métodos en {SOURCE}: {missing}. Ejecutar scripts/evaluate_all.py (sin --quick)")
    plot_nominal(methods)
    plot_impulse(methods)
    plot_regions(methods)
    plot_learning_costs(record["learning_costs"])
    plot_summary_table(methods, record["learning_costs"])
    print(f"Figuras en {FIGURES_DIR}; tabla en {TABLE_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
