# LSTM Word Oracle

A small web app around a Keras LSTM (`lstm_word_prediction.h5`) trained on the opening chapters
of *Moby-Dick*: write a seed sentence, and it predicts the next word — then keeps going,
word by word, for as many words as you ask for.

## Project layout

```
lstm_word_app/
├── app.py                       Streamlit page: seed text, controls, and the generated passage
├── requirements.txt
├── .streamlit/
│   └── config.toml              Theme and upload limit
├── assets/
│   └── style.css                All custom styling
├── models/
│   ├── lstm_word_prediction.h5  <- put your model here
│   └── tokenizer.pkl            <- required, see models/README.md
└── src/
    ├── config.py                Paths, sequence length, generation defaults
    ├── model.py                 Loads the model + tokenizer, predicts and generates
    └── ui.py                    Header, candidate-words panel, generated passage
```

## Run it

```bash
cd lstm_word_app
pip install -r requirements.txt     # skip if already installed
# copy lstm_word_prediction.h5 and tokenizer.pkl into models/
streamlit run app.py
```

Then open http://localhost:8501. If another Streamlit app is already running, add `--server.port 8502`.

## Why you need `tokenizer.pkl`

The notebook fit a Keras `Tokenizer` on the training text (`tokenizer.fit_on_texts(...)`) and
used it both to turn words into numbers for training and to turn the model's predicted numbers
back into words in `generate_text(...)`. That object was never saved — only the model was. Without
the exact same tokenizer, there's no reliable way to reproduce the same word ↔ number mapping, so
the app can't run without it. See `models/README.md` for the one-line fix.

## Two generation modes

- **Deterministic** (default) — always takes the single most likely next word. This exactly
  reproduces the notebook's `generate_text()`, which uses `np.argmax` with no randomness.
- **Creative** — samples from the model's probability distribution, reshaped by a temperature
  you control. Lower temperature stays close to the deterministic pick; higher temperature takes
  more risks. This mode isn't in the notebook; it's a small addition for more varied output.

The "oracle considered" panel always shows the top candidates for the very next word — separate
from the multi-word generation — as a window into what the model was weighing.

## Troubleshooting

- **"tokenizer.pkl not found" error.** Expected until you save it — see above.
- **Generated text looks like word salad.** Check the training notebook's final loss/accuracy
  (cell 39 in the source notebook) — 400 epochs on ~11k training sequences is a small dataset for
  a from-scratch LSTM, so fairly incoherent output is normal unless training ran long enough to
  converge further.
- **A typed word never appears in the output.** If it wasn't in the training vocabulary (only
  ~2,700 unique words in the source text), the tokenizer silently drops it — same behavior as the
  notebook's own generation function.
- **Model fails to load with a Keras error.** Install the TensorFlow version the model was
  trained with, or, for models saved with older Keras, `pip install tf-keras` and start with
  `TF_USE_LEGACY_KERAS=1 streamlit run app.py`.
