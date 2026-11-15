# Developer guide

## Architecture

```
            ┌──────────── app.py (merged app) ───────────┐
            │  sidebar radio → modules.<name>.render()   │
            └────────────────────────────────────────────┘
                     │                         │
      modules/data_loading/          modules/feature_extraction/
        module.py  (Streamlit UI)      module.py  (Streamlit UI)
        core.py    (pure logic)        core.py    (pure logic)
                     │                         │
                     └────────► bci_common/ ◄──┘
                         data.py     loaders + synthetic sample (pure)
                         plotting.py Plotly head map (pure)
                         export.py   script/JSON generation (pure)
                         ui.py       shared Streamlit widgets
```

**Rule:** anything that touches signals lives in a `core.py` or `bci_common/*.py` file **without
Streamlit imports**. That code is unit-tested and is what the exported scripts call. `module.py` only
wires widgets to `core` functions and draws figures.

### Module contract

Every module package must provide:

```python
# modules/<name>/__init__.py
from .module import render          # render() -> None, draws the whole module
MODULE_INFO = {"id": "...", "title": "...", "topic_no": 7, "icon": "🧮", "summary": "..."}
```

and should follow these conventions:

* **Widget keys** start with a module prefix (`dl_`, `fe_`), so modules never share state by accident.
* **Data source:** call `bci_common.ui.data_source_picker(prefix)`. It returns a *copy* of the Raw
  object, so the module may change it freely.
* **Explain:** call `explain_panel(markdown, questions, key=prefix)`. Each question is
  `{"q", "options", "answer" (index), "why"}`.
* **Export:** build the module-specific code `body` and call
  `export.build_script(name, source, body, params)`, then `export_panel(...)`.
* `st.set_page_config` is called **only** by the entry-point scripts (`app.py`, `modules/*/app.py`),
  never inside `render()`.

### Linked plots

Plots are linked through Plotly selection events:

1. Clickable traces carry an ID in `customdata` (channel name, trial index, feature name).
2. The chart is drawn with `st.plotly_chart(fig, key=..., on_select="rerun")`.
3. On the next run, **before any widgets are created**, `_consume(...)` reads
   `st.session_state[plot_key]`, compares it with the last selection it handled, and writes the new
   value into the state of the widget it controls (such as `fe_focus`). This order matters: Streamlit
   does not allow a widget's state to change after the widget has been created in the same run.

### Caching

* `ui.py` caches loaded recordings with `st.cache_resource` (keyed by seed, subject/run, or SHA-1 of
  the uploaded file).
* Feature extraction caches epochs (`_epochs`). The leading `_raw` argument is excluded from hashing,
  so a string signature of the source is passed alongside it.

## Merging into the shared course repository

When the course starter repository is released:

1. Copy `modules/data_loading/` and `modules/feature_extraction/` into its modules folder.
2. Copy `bci_common/` as well, or replace its imports with the starter repo's shared utilities. Only
   `data.py`, `plotting.py`, `ui.py` and `export.py` are used.
3. Register both packages in the starter app's module list. Ours is the `MODULES` list in `app.py`.
4. Merge `requirements.txt`. The only extra dependencies are those already recommended in the brief.
5. Copy `tests/` and run `pytest -q`.

If the starter repository uses a different contract (for example `render(container)` or a
`module.yaml`), add a thin adapter in `__init__.py`. The `core.py` files do not need to change.

## Adding a parameter (example)

To add a *notch filter* toggle to the data-loading module:

1. **core.py:** add a pure function, for example `notch(data, sfreq, freq)`, and a test for it.
2. **module.py:** add `sb.checkbox("Notch 50 Hz", key=f"{K}_notch")`, apply it in the computation
   block, and add it to `params` and to the exported `body`.
3. **Explain:** mention it in the markdown if it changes what the student sees.

## Tests

```
tests/test_data.py               synthetic sample: deterministic, correct units, contralateral ERD;
                                 FIF and CSV uploads; bad extensions rejected
tests/test_data_loading.py       window shape/units, alias formula, naive decimation aliases and the
                                 anti-aliased version does not, flat-channel flag, annotation filter
tests/test_feature_extraction.py band power finds a pure tone, Hjorth mobility matches frequency,
                                 feature shapes, C3/C4 rank highest, pre-cue and wrong band are near
                                 chance, CSP with shuffled labels is near chance (leakage test),
                                 chance threshold values
tests/test_export_and_apps.py    every app renders (Streamlit AppTest), every feature method renders,
                                 exported scripts run end-to-end, loader code for every source
```

Run with `pytest -q` (about 15 s). CI runs on every push (`.github/workflows/tests.yml`).

## Git workflow (recommended)

* `main` is protected. Work on feature branches (`feat/fe-band-sweep`) and merge through pull requests
  that one other team member reviews.
* Small commits with clear messages. Individual contribution marks are based on the Git history.
* Run `pytest -q` before you push.
