# Urban blue-green ranking measurement multiverse

Code and reproducibility documentation for the manuscript **“Are urban blue-green rankings properties of cities or maps? A lineage-aware measurement multiverse across 296 Chinese cities.”**

> Repository status: pre-publication research compendium. Final authors, journal citation, paper DOI, GitHub URL and archival DOI must be added before a public tagged release.

## Scientific overview

The study asks whether mapped urban blue-green rankings are stable properties of cities or conditional measurements produced by map choices. The frozen design crosses four urban-boundary scenarios from three lineages with two 2021 land-cover measurement specifications. The analysis population follows the audited cascade **296 → 294 → 285 → 284**: registered cities, cities represented in at least one setting, cities complete across all eight confirmatory settings, and complete cases in the city-characteristic models.

The repository contains the code used to:

1. build the independent city registry and measurement matrix;
2. quantify rank and leading-group instability;
3. decompose within-city variation into boundary, land-cover-specification and interaction components;
4. fit the confirmatory city-characteristic models and robustness analyses; and
5. generate the final main and supplementary figures. Supplementary Figure S1 has
   an independent Task 5 plotting script, and Supplementary Figure S2 is generated
   by the Task 6G plotting script.

The code quantifies sensitivity within the declared finite map set. It does not identify map accuracy, truth or causal effects.

## Repository structure

```text
.
├── config/                 # Frozen analysis contracts
├── data/                   # Documentation only; third-party data are not included
├── docs/                   # Data provenance, script map and reproduction guidance
├── gee/                    # Google Earth Engine export/measurement scripts
├── scripts/                # Registry, Task 5, Task 6 and figure code
├── tests/                  # Public-release integrity tests
├── CITATION.cff
├── DATA.md
├── environment.yml
├── LICENSE
├── RELEASE_MANIFEST.csv
└── requirements.txt
```

## Quick validation

```powershell
conda env create -f environment.yml
conda activate urban-blue-green-ranking
python scripts/check_release.py
python -m unittest discover -s tests -v
```

## Reproduce the analysis

Large and licensed inputs are intentionally absent. Obtain the datasets described in [DATA.md](DATA.md), recreate the directory contract in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), and then run the stage entry point:

```powershell
python scripts/reproduce_analysis.py --stage rank
python scripts/reproduce_analysis.py --stage models
python scripts/reproduce_analysis.py --stage figures
```

The residual spatial-dependence and Conley-HAC script requires ArcGIS Pro Python because it imports `arcpy`:

```powershell
& "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" scripts\run_task6f_spatial_dependence_arcpy.py
```

Google Earth Engine scripts contain `projects/YOUR_EE_PROJECT/assets`. Replace `YOUR_EE_PROJECT` with your own project and provide equivalent uploaded assets. The public package contains no personal asset identifier or credential.

Figures 1, 3 and 4 require the official 2024 administrative-map folder and, on some installations, an explicit PROJ data directory:

```powershell
$env:UBGR_ADMIN_2024_ROOT = "C:\path\to\admin2024"
$env:UBGR_PROJ_LIB = "C:\path\to\proj-data"
```

All active main and supplementary figure scripts use the shared 180-mm publication
style in `scripts/unified_figure_style.py`; figure heights remain script-specific
so multi-row layouts are not vertically distorted.

## Data and licensing

Raw third-party datasets and derived spatial assets are not redistributed. Sources, versions and access conditions are documented in [DATA.md](DATA.md). Code is released under the MIT License. Upstream dataset licences and citation requirements continue to apply to any locally generated outputs.

## Citation

I will provide the DOI of the paper after it is published.

## Contact

fanlx202@nenu.edu.cn
