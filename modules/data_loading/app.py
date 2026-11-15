"""Run this module on its own:  streamlit run modules/data_loading/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Data loading · BCI", page_icon="📂", layout="wide")
from modules.data_loading import render  # noqa: E402

render()
