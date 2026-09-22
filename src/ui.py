"""Presentation helpers: styling, the candidate-words panel, and the generated passage."""
from html import escape

import streamlit as st

from . import config
from .model import Generation


def _html(markup: str) -> str:
    return " ".join(line.strip() for line in markup.splitlines())


def inject_css() -> None:
    css = config.STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        _html(
            f"""
            <header class="masthead">
              <span class="masthead__mark">🐋</span>
              <div>
                <h1>{config.APP_TITLE}</h1>
                <p>An LSTM trained on the opening chapters of <em>{config.SOURCE_TEXT}</em>
                continues your sentence, one predicted word at a time.</p>
              </div>
            </header>
            """
        ),
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        _html(
            """
            <div class="empty">
              <strong>The page is still blank</strong>
              Write a seed sentence and press Consult the oracle.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_candidates(generation: Generation) -> None:
    if not generation.top_candidates:
        return
    rows = []
    top_prob = generation.top_candidates[0].probability or 1.0
    for i, c in enumerate(generation.top_candidates):
        pct = max(6.0, (c.probability / top_prob) * 100)
        lead = " candidate--lead" if i == 0 else ""
        rows.append(
            f'<li class="candidate{lead}">'
            f'<span class="candidate__word">{escape(c.word)}</span>'
            f'<span class="candidate__bar"><span class="candidate__fill" style="width:{pct:.1f}%"></span></span>'
            f'<span class="candidate__pct">{c.probability * 100:.1f}%</span>'
            f"</li>"
        )
    st.markdown('<p class="section-label">The oracle considered</p>', unsafe_allow_html=True)
    st.markdown(f'<ul class="candidates">{"".join(rows)}</ul>', unsafe_allow_html=True)


def render_passage(generation: Generation) -> None:
    st.markdown(
        _html(
            f"""
            <section class="page" aria-live="polite">
              <span class="page__quote">“</span>
              <p class="page__text">
                <span class="page__seed">{escape(generation.seed_text)}</span>
                <span class="page__generated"> {escape(" ".join(generation.generated_words))}</span>
                <span class="page__cursor">▍</span>
              </p>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )


def render_fineprint() -> None:
    st.markdown(
        _html(
            f"""
            <p class="fineprint">
              The vocabulary is whatever appeared in the training text ({config.SOURCE_TEXT}'s
              opening chapters) — unfamiliar words in your seed text are silently dropped before
              prediction, same as in the notebook's own generation function. Deterministic mode
              reproduces the notebook exactly; creative mode adds temperature sampling, which
              the notebook itself doesn't use.
            </p>
            """
        ),
        unsafe_allow_html=True,
    )
