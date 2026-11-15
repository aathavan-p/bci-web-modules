"""Run this module on its own:  streamlit run modules/feature_extraction/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Feature extraction · BCI", page_icon="🧮", layout="wide")
from modules.feature_extraction import render  # noqa: E402

render()
