"""Exports are runnable, and every module renders without errors."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from bci_common import export as X

ROOT = Path(__file__).resolve().parents[1]


def _run_script(code: str, tmp_path):
    f = tmp_path / "exported.py"
    f.write_text(code.replace("window.plot(", "# window.plot("))
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
    p = subprocess.run([sys.executable, str(f)], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


@pytest.mark.parametrize("app", ["app.py", "modules/data_loading/app.py",
                                 "modules/feature_extraction/app.py"])
def test_app_renders(app):
    at = AppTest.from_file(str(ROOT / app), default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) >= 5


def test_feature_methods_render():
    at = AppTest.from_file(str(ROOT / "modules/feature_extraction/app.py"),
                           default_timeout=120).run()
    for m in ("csp", "hjorth", "bandpower"):
        at.sidebar.radio(key="fe_method").set_value(m).run()
        assert not at.exception, m
        assert not at.error, m


def _export_code(at, key):
    codes = [c.value for c in at.code if "Exported from the BCI course app" in c.value]
    assert codes, f"no export code shown for {key}"
    return codes[0]


def test_exported_scripts_run(tmp_path):
    for app, key in (("modules/data_loading/app.py", "dl"),
                     ("modules/feature_extraction/app.py", "fe")):
        at = AppTest.from_file(str(ROOT / app), default_timeout=120).run()
        out = _run_script(_export_code(at, key), tmp_path)
        assert ("C x T" in out) if key == "dl" else ("CV accuracy" in out)


def test_loader_code_for_each_source():
    assert "make_synthetic_mi(seed=7)" in X.loader_code({"kind": "synthetic", "seed": 7})
    assert "load_physionet(subject=3, run=4)" in X.loader_code(
        {"kind": "physionet", "subject": 3, "run": 4})
    assert "read_raw('a.edf'" in X.loader_code({"kind": "upload", "filename": "a.edf"})
    assert "read_csv_eeg" in X.loader_code({"kind": "upload", "filename": "a.csv"})
