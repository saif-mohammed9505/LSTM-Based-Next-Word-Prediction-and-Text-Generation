"""Every setting you might need to change lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lstm_word_prediction.h5"
TOKENIZER_PATH = ROOT / "models" / "tokenizer.pkl"  # required — see models/README.md
STYLE_PATH = ROOT / "assets" / "style.css"

APP_TITLE = "LSTM Word Oracle"
SOURCE_TEXT = "Moby-Dick"

# Fallback sequence length, used only if the model doesn't report one via
# model.input_shape. The notebook trained on 25-word windows.
FALLBACK_SEQ_LEN = 25

# The exact opening line from the notebook's first training sequence — a good
# default because you can compare the app's output to the notebook's own
# generate_text(...) example.
DEFAULT_SEED_TEXT = (
    "call me ishmael some years ago never mind how long precisely having "
    "little or no money in my purse and nothing particular to interest me on"
)

WORDS_MIN, WORDS_MAX, WORDS_DEFAULT = 5, 100, 25
TOP_K = 5  # how many candidate next-words to show in the "oracle considered" panel

# Sampling. "Deterministic" always picks the single most likely next word —
# this exactly reproduces the notebook's generate_text(), which uses
# np.argmax with no randomness. "Creative" samples from the softmax
# distribution reshaped by temperature: lower temperature stays close to the
# deterministic pick, higher temperature takes more risks. This mode is not
# in the notebook — it's an optional addition for more varied output.
TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT = 0.3, 1.5, 0.8
