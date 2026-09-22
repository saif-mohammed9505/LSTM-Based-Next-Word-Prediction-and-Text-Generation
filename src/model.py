"""Loads lstm_word_prediction.h5 and its tokenizer, and predicts the next word(s)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass(frozen=True)
class Candidate:
    word: str
    probability: float


@dataclass(frozen=True)
class Generation:
    seed_text: str
    generated_words: list[str]
    full_text: str
    top_candidates: list[Candidate]  # for the very next word, before generation


class ModelService:
    def __init__(self, model_path: Path, tokenizer_path: Path):
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(tokenizer_path)

        import tensorflow as tf  # imported here so a missing install gives a clear error
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self._pad_sequences = pad_sequences
        self.path = model_path
        self.model = tf.keras.models.load_model(model_path, compile=False)

        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        self.seq_len = shape[-1] or config.FALLBACK_SEQ_LEN

        import pickle

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.index_word = self.tokenizer.index_word
        self.vocab_size = self.model.output_shape[-1]

    def _encode(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        return self._pad_sequences([encoded], maxlen=self.seq_len, truncating="pre")

    def _distribution(self, text: str) -> np.ndarray:
        probs = np.asarray(self.model.predict(self._encode(text), verbose=0))[0].astype("float64")
        probs[0] = 0.0  # index 0 is the padding token, never a real word
        total = probs.sum()
        return probs / total if total > 0 else probs

    def top_candidates(self, text: str, k: int = config.TOP_K) -> list[Candidate]:
        probs = self._distribution(text)
        top_idx = np.argsort(probs)[::-1][:k]
        return [
            Candidate(word=self.index_word.get(int(i), "?"), probability=float(probs[i]))
            for i in top_idx
            if probs[i] > 0
        ]

    def _pick_next(self, probs: np.ndarray, temperature: float | None) -> int:
        if temperature is None:
            return int(np.argmax(probs))
        # Temperature sampling: reshape the distribution, then sample.
        logits = np.log(np.clip(probs, 1e-12, None)) / temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        return int(np.random.choice(len(weights), p=weights))

    def generate(self, seed_text: str, num_words: int, temperature: float | None = None) -> Generation:
        seed_text = seed_text.strip()
        first_pass = self.top_candidates(seed_text)

        input_text = seed_text
        generated: list[str] = []
        for _ in range(num_words):
            probs = self._distribution(input_text)
            idx = self._pick_next(probs, temperature)
            word = self.index_word.get(idx)
            if word is None:  # shouldn't happen once index 0 is excluded, but stay safe
                break
            input_text = input_text + " " + word
            generated.append(word)

        return Generation(
            seed_text=seed_text,
            generated_words=generated,
            full_text=(seed_text + " " + " ".join(generated)).strip(),
            top_candidates=first_pass,
        )
