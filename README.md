# LSTM Word Oracle

A small interactive web application built with **Python, TensorFlow/Keras, LSTM, NLP, and Streamlit** for next-word prediction and text generation.

The application uses a trained LSTM model based on the opening chapters of *Moby-Dick*. Users can provide a seed sentence, and the model predicts the next word. The predicted word is then added to the input sequence and used to predict the following word, allowing the application to generate a passage word by word.

## Features

- LSTM-based next-word prediction
- Text generation from a user-provided seed sentence
- Configurable number of generated words
- Deterministic generation using the most probable next word
- Creative generation using probability sampling
- Adjustable temperature for creative generation
- Top candidate words for the next prediction
- Saved Keras model and tokenizer
- Interactive Streamlit interface
- Custom CSS styling

## How It Works

The model is trained on tokenized text from the opening chapters of *Moby-Dick*.

During generation, the application follows this process:

1. The user enters a seed sentence.
2. The sentence is converted into numerical tokens using the saved tokenizer.
3. The input sequence is prepared for the LSTM model.
4. The model predicts a probability distribution for the next word.
5. The predicted word is selected.
6. The selected word is appended to the generated text.
7. The process repeats until the requested number of words has been generated.

### Basic Workflow

`Seed Text → Tokenization → Input Sequence → LSTM → Next Word → Append Word → Repeat`

## Generation Modes

### Deterministic Mode

Deterministic mode selects the single most probable next word predicted by the model.

This reproduces the behavior of the notebook's original `generate_text()` approach, where the next word is selected using the highest-probability prediction.

The same input will therefore produce the same generated sequence.

### Creative Mode

Creative mode samples from the model's probability distribution instead of always selecting the highest-probability word.

A temperature value controls the amount of variation:

- Lower temperature produces more conservative predictions.
- Higher temperature produces more varied predictions.

Creative generation is an additional feature of the application and was not part of the original notebook generation function.

## Oracle Considered

The application includes an **Oracle Considered** panel that displays the top candidate words for the immediate next prediction.

This provides a view of the words the model considered most probable before continuing with multi-word generation.

The candidate display is separate from the complete generated passage.

## Training Data

The model was trained using text from the opening chapters of *Moby-Dick*.

The training text is processed using spaCy and converted into tokens before creating sequences for LSTM training.

The tokenizer fitted on the training text is required during inference because it maintains the mapping between words and their numerical IDs.

### Training Details

- Dataset: Opening chapters of *Moby-Dick*
- Vocabulary: Approximately 2,700 unique words
- Training sequences: Approximately 11,000
- Input sequence length: 25 words
- Training epochs: 400
- Model type: LSTM
- Task: Next-word prediction

Because the model was trained on a relatively small dataset, generated text may sometimes be repetitive or incoherent.

## Model and Tokenizer

The project uses two important files:

- `lstm_word_prediction.h5` — the trained Keras LSTM model.
- `tokenizer.pkl` — the tokenizer fitted on the original training text.

The tokenizer is essential because the LSTM model works with numerical token IDs rather than raw words.

The same tokenizer used during training must be used during prediction to preserve the original word-to-index mapping.

## Project Structure

    lstm_word_app/
    ├── app.py
    ├── requirements.txt
    ├── .streamlit/
    │   └── config.toml
    ├── assets/
    │   └── style.css
    ├── models/
    │   ├── lstm_word_prediction.h5
    │   ├── tokenizer.pkl
    │   └── README.md
    └── src/
        ├── config.py
        ├── model.py
        └── ui.py

### File Description

- `app.py` — Streamlit page containing the application interface.
- `requirements.txt` — Python dependencies required to run the application.
- `.streamlit/config.toml` — Streamlit theme and application configuration.
- `assets/style.css` — Custom application styling.
- `models/lstm_word_prediction.h5` — Trained LSTM model.
- `models/tokenizer.pkl` — Saved tokenizer required for inference.
- `models/README.md` — Information about the model files.
- `src/config.py` — Paths, sequence length, and generation defaults.
- `src/model.py` — Model and tokenizer loading, prediction, and text generation.
- `src/ui.py` — Header, candidate-word panel, and generated passage UI.

## Technologies Used

- Python
- TensorFlow
- Keras
- LSTM
- Natural Language Processing
- spaCy
- NumPy
- Streamlit
- Pickle
- CSS

## Installation

Clone the repository:

    git clone https://github.com/saif-mohammed9505/lstm-word-oracle.git

Move into the project directory:

    cd lstm-word-oracle

Install the required dependencies:

    pip install -r requirements.txt

## Model Files

Place the trained model and tokenizer inside the `models` directory:

    models/
    ├── lstm_word_prediction.h5
    └── tokenizer.pkl

The application requires both files to perform predictions.

If `tokenizer.pkl` is missing, the application cannot reliably reproduce the word-to-index mapping used when the model was trained.

## Running the Application

Start the Streamlit application with:

    streamlit run app.py

The application will normally be available at:

    http://localhost:8501

If port 8501 is already being used, run:

    streamlit run app.py --server.port 8502

## Example Workflow

A user can enter a seed sentence such as:

    call me ishmael some years ago

The application processes the input and predicts the next word.

The predicted word is appended to the sentence, and the updated text is passed back through the model.

This continues repeatedly:

`Seed Sentence → Predict → Append → Predict → Append → Generate`

The user can control how many words should be generated.

## Deterministic Generation

In deterministic mode, the application selects the word with the highest predicted probability.

Conceptually:

    next_word = argmax(predicted_probabilities)

This means the model always selects the most probable candidate at each step.

For the same input and model, deterministic generation produces the same sequence.

## Creative Generation

In creative mode, the application samples from the predicted probability distribution rather than always selecting the highest-probability word.

Temperature controls the distribution used for sampling:

- Lower temperature makes the output more predictable.
- Higher temperature increases variation.
- Very high temperatures can produce less coherent text.

Creative generation allows the same seed text to produce different outputs.

## Oracle Considered

Before generating the complete passage, the application can display the top candidate words considered for the next prediction.

For example:

| Rank | Candidate Word |
|---|---|
| 1 | word A |
| 2 | word B |
| 3 | word C |
| 4 | word D |
| 5 | word E |

This gives the user an idea of what the model considers likely for the immediate next word.

## Limitations

The model is a relatively small LSTM trained on a limited amount of text.

Because of this, longer generated passages may sometimes become:

- Repetitive
- Grammatically inconsistent
- Semantically inconsistent
- Incoherent

The quality of generated text depends on the training data, model architecture, training process, and input sequence.

Words that are not represented in the training vocabulary may not be handled in the same way as words seen during training.

The model should therefore be viewed as a demonstration of LSTM-based language modeling rather than a modern large-scale language model.

## Troubleshooting

### `tokenizer.pkl not found`

Make sure `tokenizer.pkl` is present inside:

    models/tokenizer.pkl

The tokenizer must correspond to the tokenizer used when training the LSTM model.

### Generated Text Looks Like Word Salad

The model was trained on a relatively small text dataset using a from-scratch LSTM architecture.

The training dataset contains approximately 11,000 training sequences and approximately 2,700 unique words.

As a result, the model may not learn enough language patterns to generate consistently coherent long passages.

Check the final training loss and accuracy from the original training notebook when evaluating the model.

### Input Word Does Not Appear in the Generated Text

A word that was not included in the training vocabulary cannot be represented by the tokenizer in the same way as a known word.

The tokenizer used during training should always be reused during inference.

### Model Loading Error

If the saved model produces a Keras or TensorFlow compatibility error, use a compatible TensorFlow/Keras environment matching the environment in which the model was trained.

For models saved with older Keras versions, legacy Keras compatibility may also be required.

One possible legacy setup is:

    pip install tf-keras

Then, if required by the model environment, start Streamlit with:

    TF_USE_LEGACY_KERAS=1 streamlit run app.py

## Learning Objectives

This project demonstrates:

- LSTM networks for sequential data
- NLP text preprocessing
- Text tokenization
- Word-to-index conversion
- Sequence generation for language modeling
- Next-word prediction
- Training a neural language model
- Saving and loading a trained Keras model
- Saving and reusing a tokenizer
- Iterative text generation
- Probability-based word sampling
- Temperature-based generation
- Building a Streamlit interface around a deep learning model

## Project Type

**Deep Learning | NLP | LSTM | Next-Word Prediction | Text Generation | Streamlit**
## 📌 Repository

GitHub Repository:

https://github.com/saif-mohammed9505/LSTM-Based-Next-Word-Prediction-and-Text-Generation

---

## 👨‍💻 Author

**Saif Mohammed**

GitHub:

https://github.com/saif-mohammed9505

---

## License

This project is intended for educational and experimental purposes.
