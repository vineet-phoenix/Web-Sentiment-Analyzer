import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
from textblob import Word
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re # For regex operations
import os
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Re-download NLTK resources if not already present (safe to run)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Define paths for the saved model and tokenizer
save_path = '/content/drive/MyDrive/my_models'
model_path = os.path.join(save_path, 'emotion_detection_model_epoch_05.h5')
tokenizer_file_path = os.path.join(save_path, 'tokenizer.json')

# Load the tokenizer
loaded_tokenizer = None
if os.path.exists(tokenizer_file_path):
    with open(tokenizer_file_path, 'r') as f:
        loaded_tokenizer_json = json.load(f)
    loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)
    print("Tokenizer loaded successfully!")
else:
    print(f"Error: Tokenizer file not found at {tokenizer_file_path}. Cannot make prediction.")

# Load the saved model
loaded_model = None
try:
    loaded_model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}. Cannot make prediction.")

# Max_len was determined during the tokenizer fitting step (e.g., 75)
# If you run the tokenizer fitting cell again, ensure this value is consistent.
# For now, we'll hardcode the value observed from the previous run.
max_len = 75 # Based on the output from cell z51kdn9thlBF

# Define the emotion categories mapping
emotion_categories = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def predict_emotion(input_text):
    """
    Predicts the emotion of an input string using the trained model and loaded tokenizer.

    Args:
        input_text (str): The text string for emotion prediction.

    Returns:
        str: The predicted emotion category (sadness, joy, love, anger, fear, surprise),
             or an error message if the model or tokenizer is not loaded.
    """
    if loaded_model is None or loaded_tokenizer is None:
        return "Error: Model or Tokenizer not loaded. Cannot make prediction."

    # Preprocess the input text, matching the training pipeline
    processed_text = input_text.lower()
    processed_text = re.sub(r'\d+', '', processed_text) # Remove numbers

    words_in_text = processed_text.split() # Split by whitespace

    # Remove stop words
    words_in_text = [word for word in words_in_text if word not in stop_words]

    # Lemmatize words
    lemmatized_words = [Word(word).lemmatize() for word in words_in_text]
    processed_text = ' '.join(lemmatized_words)

    # Tokenize and pad the preprocessed text using the LOADED tokenizer
    seq = loaded_tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=max_len) # Use max_len from training data

    # Make prediction
    prediction = loaded_model.predict(padded_seq, verbose=0) # verbose=0 to suppress output for single prediction
    predicted_label_index = np.argmax(prediction)

    # Return the associated category
    return emotion_categories.get(predicted_label_index, "Unknown Category")

# Example Usage:
print("\n--- Testing the predict_emotion function ---")
print(f"'I am so happy to see you!' -> {predict_emotion('I am so happy to see you!')}")
#print(f"'This is making me incredibly sad.' -> {predict_emotion('This is making me incredibly sad.')}")
#print(f"'I absolutely adore this new song!' -> {predict_emotion('I absolutely adore this new song!')}")
#print(f"'I am filled with rage over this decision.' -> {predict_emotion('I am filled with rage over this decision.')}")
#print(f"'A sudden loud noise startled me.' -> {predict_emotion('A sudden loud noise startled me.')}")
#print(f"'I was surprised by the unexpected gift.' -> {predict_emotion('I was surprised by the unexpected gift.')}")
#print(f"'I feel nothing right now.' -> {predict_emotion('I feel nothing right now.')}")
