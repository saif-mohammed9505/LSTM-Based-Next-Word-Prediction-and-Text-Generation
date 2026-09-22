Put your trained model here as `lstm_word_prediction.h5`, plus:

- `tokenizer.pkl` — **required**. The notebook never saved the Keras
  `Tokenizer` it fit on the training text, but the app needs it to turn typed
  words into the same integer indices the model was trained on, and to turn
  predicted indices back into words. While the notebook kernel still has
  `tokenizer` in memory, run:

  ```python
  import pickle
  with open("tokenizer.pkl", "wb") as f:
      pickle.dump(tokenizer, f)
  ```

  then copy the resulting file here. Without it, the app can't run — there's
  no way to reconstruct the exact same word-to-number mapping otherwise.

To use different file names, change `MODEL_PATH` / `TOKENIZER_PATH` in
`src/config.py`.
