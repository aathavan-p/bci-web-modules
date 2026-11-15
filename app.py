"""BCI course app - entry point that hosts every teaching module.

Run:  streamlit run app.py
"""
import importlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Register modules here: (python package, ...). Each must expose render() and MODULE_INFO.
MODULES = [
    "modules.data_loading",
    "modules.feature_extraction",
]

st.set_page_config(page_title="BCI course app", page_icon="🧠", layout="wide")

loaded = [importlib.import_module(m) for m in MODULES]
infos = [m.MODULE_INFO for m in loaded]

qp = st.query_params.get("module")
ids = [i["id"] for i in infos]
default = ids.index(qp) if qp in ids else 0

st.sidebar.title("🧠 BCI course app")
choice = st.sidebar.radio(
    "Module", range(len(loaded)), index=default,
    format_func=lambda i: f"{infos[i]['icon']} {infos[i]['topic_no']}. {infos[i]['title']}",
    key="module_choice")
st.query_params["module"] = ids[choice]
st.sidebar.caption(infos[choice]["summary"])
st.sidebar.divider()

loaded[choice].render()
