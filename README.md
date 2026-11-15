# BCI teaching modules: Data loading & Feature extraction

Interactive web modules for the **Brain–Computer Interfaces (CS5109)** course project, **Option A**.
They teach two steps of the BCI pipeline on real or realistic EEG:

| # | Module | What a student learns |
|---|--------|-----------------------|
| 1 | **📂 Data loading** | What is inside an EEG file: the C × T matrix, sampling rate and Nyquist, units (V vs µV), montage, annotations → events. Includes a live aliasing demo and a per-channel quality check. |
| 7 | **🧮 Feature extraction** | How a trial (channels × samples) becomes a feature vector: band power (Welch), CSP log-variance, Hjorth parameters. Which channels and bands carry class information, and how to check that without data leakage. |

Built with **Streamlit + MNE-Python + Plotly + scikit-learn**, as recommended in the brief.

---

## Quick start (one-line run)

macOS / Linux / Windows Command Prompt / PowerShell 7+:

```bash
pip install -r requirements.txt && streamlit run app.py
```

Windows PowerShell 5.1 (the default one), which does not accept `&&`:

```powershell
python -m pip install -r requirements.txt; python -m streamlit run app.py
```

Already installed? Start it with `streamlit run app.py` (or `python -m streamlit run app.py`).

The browser opens at <http://localhost:8501>. Pick a module in the sidebar.

Run one module on its own:

```bash
streamlit run modules/data_loading/app.py
streamlit run modules/feature_extraction/app.py
```

Run the tests:

```bash
pytest -q
```

With conda: `conda env create -f environment.yml && conda activate bci-web-modules`.

---

## Brief requirements → where they are met

| Required by the brief | Data loading | Feature extraction |
|---|---|---|
| Sample data + own file upload | Synthetic sample (offline), PhysioNet EEGBCI, upload `.fif .edf .bdf .gdf .set .csv` | same picker |
| 5+ adjustable parameters | window start, window length, channels, focus channel, amplitude spacing, resample rate, anti-alias on/off, annotation labels, head-map statistic | classes, epoch window, channel set, method, band preset, band range, Welch segment, relative power, log transform, CSP components, Hjorth parameter |
| 2 linked, live-updating plots | traces ↔ head map ↔ spectrum ↔ navigator. Click a sensor to add it and focus its spectrum. Click an event to jump there. | class spectra ↔ separability head map ↔ feature scatter ↔ Fisher bars. Click a sensor to change the spectrum and scatter axis. Click a trial to overlay its spectrum. Click a bar to plot that feature. |
| “Explain” panel with questions | Live text that uses the current values (samples in the window, Nyquist, where 50 Hz aliases to) and 5 self-check questions | Live text per method (frequency resolution, CSP and leakage, chance threshold) and 5 self-check questions |
| Export of code and parameters | Runnable `.py`, `params.json`, window as CSV | Runnable `.py`, `params.json`, `features.csv` |
| One-line run command | `streamlit run app.py` | same |
| At least 3 automated tests | **24 tests** in `tests/` (signal correctness, leakage, UI rendering, and checks that exported scripts run). GitHub Actions runs them on every push. | |

---

## Data sources

* **Sample (synthetic motor imagery).** Generated from a fixed seed, works offline, 26 channels at 160 Hz,
  20 trials each of `left_hand`, `right_hand` and `rest`. The ground truth is known: occipital alpha,
  mu/beta over C3/C4 with *contralateral* desynchronisation that varies from trial to trial, eye blinks at
  Fp1/Fp2, and 50 Hz mains noise. Students can check what they see against what is known to be there.
* **PhysioNet EEG Motor Movement/Imagery (EEGBCI).** Real 64-channel EEG from 109 subjects, downloaded on
  first use through `mne.datasets.eegbci` (about 2–3 MB per run, saved to `~/mne_data`). Runs 4, 8 and 12
  are imagined left/right fist.
* **Upload.** Single-file formats MNE can read. For CSV, use one column per channel (named with 10-20
  labels if you want head maps), values in µV, and an optional `time` column. You set the sampling rate
  in the app.

## Repository layout

```
app.py                         # merged course app (sidebar module switcher)
bci_common/                    # shared by all modules
  data.py                      #   loaders, synthetic generator, helpers (no Streamlit)
  plotting.py                  #   Plotly head map
  ui.py                        #   data-source picker, Explain panel, export panel
  export.py                    #   script / JSON generation
modules/
  data_loading/
    __init__.py                #   exposes render() and MODULE_INFO
    core.py                    #   signal logic (pure, tested)
    module.py                  #   Streamlit UI
    app.py                     #   standalone entry point
  feature_extraction/          #   same layout
tests/                         # pytest suite
docs/                          # user guide, developer guide, contribution statement, AI-use note
.github/workflows/tests.yml    # CI
```

Each module lives in its own folder and exposes a single `render()` function, as the brief asks. See
[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for how to merge into the shared course repository.

## Documentation

* [User guide](docs/USER_GUIDE.md): a walkthrough of each module, with suggested experiments
* [Developer guide](docs/DEVELOPER_GUIDE.md): architecture, adding parameters, merging, testing
* [Contribution statement template](docs/CONTRIBUTION_STATEMENT.md)
* [10-minute demo script](docs/DEMO_SCRIPT.md)
* [AI-use statement](docs/AI_USE.md)

## References

* Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
* Schalk, G. et al. (2004). BCI2000: a general-purpose brain-computer interface system. *IEEE TBME*, 51(6). Dataset: Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit and PhysioNet. *Circulation*, 101(23).
* Pfurtscheller, G. & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11).
* Ramoser, H., Müller-Gerking, J. & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE TRE*, 8(4).
* Hjorth, B. (1970). EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*, 29(3).
* Combrisson, E. & Jerbi, K. (2015). Exceeding chance level by chance. *Journal of Neuroscience Methods*, 250.

## Team

| Name | Roll no. | Main responsibility |
|------|----------|---------------------|
| Aathavan P | ME23B1028 | Shared data layer (`bci_common/data.py`): synthetic motor-imagery sample generator, PhysioNet EEGBCI loader, file upload (FIF/EDF/BDF/GDF/SET/CSV); repository setup and merge into the course app (`app.py`) |
| Harivarman M | ME23B1025 | Data Loading module (`modules/data_loading/`): window extraction, resampling and aliasing demo, channel quality statistics; linked traces, head map, spectrum and navigator plots; Explain panel and questions |
| Nishal V | ME23B1026 | Feature extraction core (`modules/feature_extraction/core.py`): band power (Welch), CSP, Hjorth parameters, Fisher score, leak-free cross-validation and chance threshold |
| Pazhaniyappan P | ME23B2003 | Feature Extraction UI (`modules/feature_extraction/module.py`): linked class spectra, separability head map, feature scatter and Fisher bars, band sweep; Explain panel and questions |
| Siva BalaKumaran S | ME23B1069 | Shared UI and quality (`bci_common/ui.py`, `plotting.py`, `export.py`): code and parameter export, head-map plotting; automated tests and CI; README, user and developer guides |
