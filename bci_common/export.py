"""Helpers that turn the current UI state into reproducible artefacts:
a JSON parameter file and a standalone Python script."""
from __future__ import annotations

import json
from datetime import datetime, timezone


def loader_code(source: dict) -> str:
    """Python lines that recreate ``raw`` for a given data-source dict.

    ``source`` keys: ``kind`` in {synthetic, physionet, upload} and the
    parameters of that kind.
    """
    kind = source.get("kind")
    if kind == "synthetic":
        return (
            "# Synthetic motor-imagery sample shipped with this repo\n"
            "from bci_common.data import make_synthetic_mi\n"
            f"raw = make_synthetic_mi(seed={int(source.get('seed', 42))})\n"
        )
    if kind == "physionet":
        return (
            "# PhysioNet EEG Motor Movement/Imagery dataset (downloads on first run)\n"
            "from bci_common.data import load_physionet\n"
            f"raw = load_physionet(subject={int(source['subject'])}, run={int(source['run'])})\n"
        )
    if kind == "upload":
        fname = source.get("filename", "my_recording.fif")
        if fname.lower().endswith(".csv"):
            return (
                "# Your CSV file: one column per channel, values in microvolts\n"
                "from bci_common.data import read_csv_eeg\n"
                f"raw = read_csv_eeg({fname!r}, sfreq={float(source.get('csv_sfreq', 250.0))})\n"
            )
        return (
            "# Your own recording (put the file next to this script)\n"
            "import mne\n"
            "from bci_common.data import try_set_montage\n"
            f"raw = mne.io.read_raw({fname!r}, preload=True)\n"
            "try_set_montage(raw)\n"
        )
    raise ValueError(f"Unknown source kind: {kind!r}")


def build_script(module_name: str, source: dict, body: str, params: dict) -> str:
    """Wrap module-specific ``body`` code into a runnable script."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f'"""Exported from the BCI course app - module: {module_name}\n'
        f"Generated {stamp}.\n\n"
        "Run from the repository root:  python this_script.py\n"
        f"Parameters used:\n{json.dumps(params, indent=2, default=str)}\n"
        '"""\n'
        "import sys, os\n"
        "sys.path.insert(0, os.getcwd())  # so `bci_common` is importable\n"
        "import mne\nimport numpy as np\n"
        "mne.set_log_level('ERROR')\n\n"
    )
    return header + loader_code(source) + "\n" + body.strip() + "\n"


def params_json(module_name: str, source: dict, params: dict) -> str:
    return json.dumps({"module": module_name, "source": source, "params": params},
                      indent=2, default=str)
