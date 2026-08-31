import csv
import hashlib
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_checker():
    path = ROOT / "scripts" / "check_release.py"
    spec = importlib.util.spec_from_file_location("check_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleaseIntegrityTests(unittest.TestCase):
    def test_release_checker_passes(self):
        self.assertEqual(load_checker().audit(), [])

    def test_core_workflow_inventory_is_present(self):
        required = (
            "scripts/build_task5_ranking_instability.py",
            "scripts/decompose_task5_map_sensitivity.py",
            "scripts/run_task6_confirmatory_model.py",
            "scripts/run_task6_component_models_fdr.py",
            "scripts/plot_r2_figure1_study_area_measurement_scenarios.py",
            "scripts/plot_task5_publication_figures.py",
            "scripts/plot_r2_spatial_sensitivity_figures.py",
            "scripts/plot_r2_figure5_city_characteristics.py",
        )
        for relative in required:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_manifest_hashes_match(self):
        manifest = ROOT / "RELEASE_MANIFEST.csv"
        self.assertTrue(manifest.exists())
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 50)
        for row in rows:
            path = ROOT / row["path"]
            self.assertTrue(path.exists(), path)
            self.assertEqual(int(row["bytes"]), path.stat().st_size)
            self.assertEqual(row["sha256"], sha256(path))

    def test_release_contains_no_spatial_data_files(self):
        forbidden = {".shp", ".gpkg", ".tif", ".tiff", ".gdb", ".parquet", ".feather"}
        found = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
