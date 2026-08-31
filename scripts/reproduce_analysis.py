from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STAGES = {
    "rank": [
        "audit_task5_ranking_frame.py",
        "build_task5_ranking_instability.py",
        "decompose_task5_map_sensitivity.py",
        "run_task5_stage3_sensitivity.py",
        "run_task5_lineage_balance_sensitivity.py",
    ],
    "models": [
        "build_task6_model_frame_and_premodel_diagnostics.py",
        "run_task6_confirmatory_model.py",
        "run_task6_component_models_fdr.py",
        "run_task6e_robustness.py",
    ],
    "figures": [
        "build_r2_figure_source_data.py",
    "plot_r2_figure1_study_area_measurement_scenarios.py",
    "plot_task5_publication_figures.py",
    "plot_task5_supplementary_figure_s1.py",
    "plot_r2_spatial_sensitivity_figures.py",
        "plot_r2_figure5_city_characteristics.py",
        "plot_task6g_publication_figures.py",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen analysis stages after required inputs are installed.")
    parser.add_argument("--stage", choices=["rank", "models", "figures", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the execution order without running scripts.")
    args = parser.parse_args()
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in selected:
        for name in STAGES[stage]:
            command = [sys.executable, str(SCRIPTS / name)]
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
