"""Stand-alone FermenTrack Model Workbench.

Run with:  streamlit run workbench_app.py      (or double-click Workbench.bat)
The same workbench is also a page inside the main dashboard (app.py).
"""
import streamlit as st

from fermentrack.workbench.ui import render

st.set_page_config(page_title="FermenTrack Workbench", page_icon="🔬", layout="wide")
render(standalone=True)
