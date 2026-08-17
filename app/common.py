"""Shared boilerplate for every page in the reviewer dashboard (Introduction.py + pages/*.py) -
the bit that would otherwise be duplicated across every page: page config and cached file reads.
Page-specific content (which artifact, which numbers, which words) stays in the page itself.
"""
import json

import pandas as pd
import streamlit as st


def page_config(title: str) -> None:
    st.set_page_config(page_title=f"{title} - Fresh-Food Ordering", layout="wide")
    st.title(title)


@st.cache_data
def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)
