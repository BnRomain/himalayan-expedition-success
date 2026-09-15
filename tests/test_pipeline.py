"""End-to-end tests of the five scripts.

The scripts are procedural, so each test runs one of them in-process (which keeps
the coverage measurement) and checks its outputs: the printed facts, the files
written to data/ and the figures written to docs/figures/.
"""

import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
FIGURES = ROOT / "docs" / "figures"

FEATURES_BASE = ["o2used", "comrte", "tothired", "totmembers", "camps", "year"]
ONE_HOT = [
    "season_Spring",
    "season_Summer",
    "season_Winter",
    "peakid_Autre",
    "peakid_BARU",
    "peakid_CHOY",
    "peakid_DHA1",
    "peakid_EVER",
    "peakid_LHOT",
    "peakid_MAKA",
    "peakid_MANA",
]


def run(script, capsys):
    runpy.run_path(str(SRC / script), run_name="__main__")
    return capsys.readouterr().out


def percent(pattern, text):
    match = re.search(pattern, text)
    assert match, f"{pattern!r} not found in the output"
    return float(match.group(1)) / 100


def test_exploration_reduces_the_raw_dataset_to_21_columns(capsys):
    out = run("01_exploration.py", capsys)
    assert "Dataset brut : 11425 lignes, 65 colonnes" in out
    assert "Dataset reduit : 11425 lignes, 21 colonnes" in out
    assert "Shape : (11425, 9)" in out


def test_wrangling_writes_the_clean_dataset(capsys):
    out = run("02_wrangling.py", capsys)
    assert "Lignes retirees (nettoyage) : 63  ->  reste 11362" in out

    depart = pd.read_csv(DATA / "exped_depart.csv")
    assert depart.shape == (11425, 21)

    clean = pd.read_csv(DATA / "exped_clean.csv")
    assert clean.shape == (11362, 19)
    assert list(clean.columns) == FEATURES_BASE + ["success", "ratio_hired"] + ONE_HOT
    assert not clean.isnull().any().any()
    assert set(clean["success"].unique()) == {0, 1}
    assert set(clean["o2used"].unique()) == {0, 1}
    # One reference category dropped per variable: at most one season and one peak per row.
    assert clean[[c for c in ONE_HOT if c.startswith("season_")]].sum(axis=1).max() == 1
    assert clean[[c for c in ONE_HOT if c.startswith("peakid_")]].sum(axis=1).max() == 1
    # The Sherpa ratio is defined for every row (no expedition without members left).
    assert (clean["totmembers"] > 0).all()
    assert np.allclose(clean["ratio_hired"], clean["tothired"] / clean["totmembers"])
    # The target stays balanced after cleaning.
    assert 0.5 < clean["success"].mean() < 0.6


@pytest.mark.parametrize(
    ("script", "figures"),
    [
        (
            "03_visualisation.py",
            [
                "01_effet_oxygene.png",
                "02_interaction_o2_saison.png",
                "03_sherpas_vs_membres.png",
                "04_correlations.png",
                "08_oxygene_par_sommet.png",
            ],
        ),
        ("04_feature_engineering.py", ["05_ratio_hired.png", "06_rope_bool.png"]),
    ],
)
def test_visualisation_scripts_write_their_figures(script, figures, capsys):
    for name in figures:
        (FIGURES / name).unlink(missing_ok=True)
    run(script, capsys)
    for name in figures:
        assert (FIGURES / name).stat().st_size > 0


def test_feature_engineering_proves_the_confounding(capsys):
    out = run("04_feature_engineering.py", capsys)
    everest_o2 = percent(r"utilisent l'O2 : (\d+)%", out)
    everest_spring = percent(r"faites au printemps : (\d+)%", out)
    everest_share = percent(r"'Printemps\+O2' : (\d+)%", out)
    assert everest_o2 > 0.7
    assert everest_spring > 0.8
    assert everest_share > 0.6
    assert "ratio_hired  -> GARDEE" in out
    assert "saison_o2    -> REJETEE" in out
    assert "rope_bool    -> REJETEE" in out


def test_regression_beats_the_baseline(capsys):
    run("02_wrangling.py", capsys)
    capsys.readouterr()
    for name in ("07_matrice_confusion.png", "09_coefficients.png"):
        (FIGURES / name).unlink(missing_ok=True)

    out = run("05_regression.py", capsys)

    # camps is excluded (data leakage): 19 columns, minus the target and camps.
    assert "Dataset : 11362 expeditions, 17 predicteurs" in out
    baseline = percent(r"Baseline \(classe majoritaire\) : ([\d.]+)%", out)
    holdout = percent(r"Holdout \(split 80/20\)\s+: accuracy = ([\d.]+)%", out)
    cv = percent(r"Validation croisee \(5 plis\): accuracy = ([\d.]+)%", out)
    assert 0.5 < baseline < 0.6
    assert holdout > baseline + 0.1
    assert cv > baseline + 0.1
    assert abs(holdout - cv) < 0.03

    # Oxygen and the commercial route are the strongest positive coefficients.
    coefs = dict(re.findall(r"^(\w+)\s+(-?\d+\.\d+)$", out, flags=re.MULTILINE))
    assert float(coefs["o2used"]) > 1.5
    assert float(coefs["comrte"]) > 1.0
    assert float(coefs["peakid_EVER"]) < -1.0
    assert "camps" not in coefs

    for name in ("07_matrice_confusion.png", "09_coefficients.png"):
        assert (FIGURES / name).stat().st_size > 0
