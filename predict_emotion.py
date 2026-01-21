import tensorflow as tf
import numpy as np
import nltk
from textblob import Word
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import os
import json
import argparse
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Re-download NLTK resources if not already present (safe to run)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Default filenames (change if your files have different names)
DEFAULT_MODEL_FILENAME = 'emotion_detection_model_epoch_05.h5'
DEFAULT_TOKENIZER_FILENAME = 'tokenizer.json'

# Max_len used in training (ensure this matches your training)
max_len = 75

# Define the emotion categories mapping
emotion_categories = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def resolve_model_dir(cli_dir=None):
    """
    Resolve the model directory in this order:
    1. CLI-provided directory
    2. Environment variable MODEL_DIR
    3. models directory next to this script (repo-root/models/)
    """
    candidates = []
    if cli_dir:
        candidates.append(os.path.abspath(cli_dir))
    env_dir = os.getenv('MODEL_DIR')
    if env_dir:
        candidates.append(os.path.abspath(env_dir))

    # models folder next to script (recommended for repo layout)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, 'models'))

    # models in cwd as a fallback
    candidates.append(os.path.join(os.getcwd(), 'models'))

    for d in candidates:
        if d and os.path.exists(d):
            return d

    # If none exist, return the script-local models path (where you should place files)
    return os.path.join(script_dir, 'models')

def load_tokenizer_and_model(model_dir, model_filename=DEFAULT_MODEL_FILENAME, tokenizer_filename=DEFAULT_TOKENIZER_FILENAME):
    """
    Attempts to load tokenizer and model from model_dir.
    Returns (loaded_model, loaded_tokenizer).
    """
    if model_dir is None:
        print("No model directory provided/found.")
        return None, None

    model_path = os.path.join(model_dir, model_filename)
    tokenizer_path = os.path.join(model_dir, tokenizer_filename)

    loaded_tokenizer = None
    loaded_model = None

    # Load tokenizer
    if os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                loaded_tokenizer_json = json.load(f)
            loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)
            print(f"Tokenizer loaded successfully from {tokenizer_path}")
        except Exception as e:
            print(f"Error loading tokenizer from {tokenizer_path}: {e}")
    else:
        print(f"Tokenizer file not found at {tokenizer_path}")

    # Load model
    if os.path.exists(model_path):
        try:
            loaded_model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    else:
        print(f"Model file not found at {model_path}")

    return loaded_model, loaded_tokenizer

def predict_emotion(input_text, loaded_model, loaded_tokenizer):
    """
    Predicts the emotion of an input string using the trained model and loaded tokenizer.
    """
    if loaded_model is None or loaded_tokenizer is None:
        return "Error: Model or Tokenizer not loaded. Cannot make prediction."

    # Preprocess the input text (match training pipeline)
    processed_text = input_text.lower()
    processed_text = re.sub(r'\d+', '', processed_text)  # Remove numbers

    words_in_text = processed_text.split()
    words_in_text = [word for word in words_in_text if word not in stop_words]

    lemmatized_words = [Word(word).lemmatize() for word in words_in_text]
    processed_text = ' '.join(lemmatized_words)

    # Tokenize and pad the preprocessed text
    seq = loaded_tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=max_len)

    # Make prediction
    prediction = loaded_model.predict(padded_seq, verbose=0)
    predicted_label_index = np.argmax(prediction)

    return emotion_categories.get(predicted_label_index, "Unknown Category")

def main():
    parser = argparse.ArgumentParser(description="Load emotion detection model and tokenizer and run examples")
    parser.add_argument('--model-dir', help='Directory containing the model (.h5) and tokenizer.json', default=None)
    parser.add_argument('--model-file', help='Model filename (default emotion_detection_model_epoch_05.h5)', default=DEFAULT_MODEL_FILENAME)
    parser.add_argument('--tokenizer-file', help='Tokenizer filename (default tokenizer.json)', default=DEFAULT_TOKENIZER_FILENAME)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    print(f"Using model directory: {model_dir}")

    loaded_model, loaded_tokenizer = load_tokenizer_and_model(model_dir, args.model_file, args.tokenizer_file)

    # Example Usage:
    #print("\n--- Testing the predict_emotion function ---")
    #test_text = "I am so happy to see you!"
    #print(f"'{test_text}' -> {predict_emotion(test_text, loaded_model, loaded_tokenizer)}")

if __name__ == '__main__':
    main()
