"""Rutas del repositorio. El paquete se instala en modo editable (`pip install -e .`),
así que la raíz del repo se deduce de la ubicación de este archivo (src/cartpole_lab/)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TUNING_DIR = RESULTS_DIR / "tuning"  # parámetros sintonizados de los controladores clásicos
WEIGHTS_DIR = RESULTS_DIR / "weights"  # pesos de las redes (Fase 2)
