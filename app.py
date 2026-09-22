"""Streamlit entry point.  Run with:  streamlit run app.py"""
import streamlit as st

from src import config, ui
from src.model import ModelService

st.set_page_config(page_title=config.APP_TITLE, page_icon="🐋", layout="wide")
ui.inject_css()


@st.cache_resource(show_spinner="Loading model and tokenizer…")
def get_service() -> ModelService:
    return ModelService(config.MODEL_PATH, config.TOKENIZER_PATH)


def load_service_or_stop() -> ModelService:
    try:
        return get_service()
    except FileNotFoundError as exc:
        missing = exc.args[0]
        if str(missing).endswith("tokenizer.pkl"):
            st.error(
                "`tokenizer.pkl` not found in `models/`. The notebook never saved the "
                "Tokenizer, so this app can't turn words into the numbers the model expects. "
                "In the notebook, run:\n\n"
                "```python\nimport pickle\nwith open('tokenizer.pkl', 'wb') as f:\n"
                "    pickle.dump(tokenizer, f)\n```\n\nthen copy that file into `models/`."
            )
        else:
            st.error(
                f"Model file not found. Put `lstm_word_prediction.h5` in the `models/` folder "
                f"(expected at `{config.MODEL_PATH}`), then reload."
            )
    except ImportError:
        st.error("TensorFlow isn't installed. Run `pip install -r requirements.txt`, then restart.")
    except Exception as exc:  # noqa: BLE001 - show any load failure to the user
        st.error(f"The model couldn't be loaded: {exc}")
    st.stop()


def main() -> None:
    service = load_service_or_stop()

    with st.sidebar:
        st.subheader("Model")
        st.caption(f"File: {service.path.name}")
        st.caption(f"Sequence length: {service.seq_len} words")
        st.caption(f"Vocabulary: {service.vocab_size:,} tokens")
        st.divider()
        st.subheader("Generation")
        num_words = st.slider("Words to generate", config.WORDS_MIN, config.WORDS_MAX, config.WORDS_DEFAULT)
        creative = st.toggle("Creative mode", value=False, help="Samples from the probability distribution instead of always taking the top pick.")
        temperature = None
        if creative:
            temperature = st.slider(
                "Temperature",
                config.TEMPERATURE_MIN, config.TEMPERATURE_MAX, config.TEMPERATURE_DEFAULT, step=0.05,
                help="Lower stays close to the model's top choice; higher takes more risks.",
            )
        else:
            st.caption("Deterministic — reproduces the notebook's generate_text() exactly.")

    ui.render_header()

    st.markdown('<p class="section-label">Seed text</p>', unsafe_allow_html=True)
    seed_text = st.text_area(
        "Seed text", value=config.DEFAULT_SEED_TEXT, height=110, label_visibility="collapsed"
    )
    go = st.button("🔮 Consult the oracle", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not go:
        ui.render_empty_state()
    elif not seed_text.strip():
        st.warning("Write a seed sentence first.")
    else:
        with st.spinner("Weighing the vocabulary…"):
            generation = service.generate(seed_text, num_words, temperature)
        ui.render_passage(generation)
        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_candidates(generation)

    ui.render_fineprint()


main()
