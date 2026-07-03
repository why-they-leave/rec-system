from pathlib import Path

import streamlit as st


def load_css(path: str = "app/static/style.css") -> None:
    """CSS 파일을 읽어 st.markdown으로 주입. main.py 또는 각 페이지 상단에서 1회 호출."""
    css_path = Path(path)
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
