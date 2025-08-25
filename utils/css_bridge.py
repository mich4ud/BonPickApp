# utils/css_bridge.py
from __future__ import annotations
from pathlib import Path
import streamlit as st

def inject_css(css_text: str):
    """Injecte un bloc CSS brut dans Streamlit."""
    if not css_text:
        return
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

def inject_css_file(path: str | Path):
    """Charge et injecte un fichier CSS si présent."""
    p = Path(path)
    if p.exists():
        try:
            inject_css(p.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Impossible de charger {p.name} : {e}")

def save_css(path: str | Path, css_text: str):
    """Écrit/écrase un fichier CSS."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(css_text, encoding="utf-8")
