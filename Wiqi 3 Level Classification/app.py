import streamlit as st
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

save_dir = 'model_components'

# Load the tokenizer
with open(os.path.join(save_dir, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

# Load max_length
with open(os.path.join(save_dir, 'max_length.pkl'), 'rb') as f:
    max_length = pickle.load(f)

with open(os.path.join(save_dir, 'stop_words.pkl'), 'rb') as f:
    stop_words = pickle.load(f)
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    st_words = stop_words
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text
def classify_text(text):
    text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = np.pad(seq, ((0, 0), (0, max_length - len(seq[0]))), mode='constant')

    prediction = model.predict(padded_seq)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    categories = predicted_label.split('|')
    
    if len(categories) == 3:
        main_category = categories[0]
        sub_category = categories[1]
        lowest_category = categories[2]
    else:
        main_category = "Unknown"
        sub_category = "Unknown"
        lowest_category = "Unknown"
    print(f"Main Category: {main_category}, Sub Category: {sub_category}, Lowest Category: {lowest_category}")

# Load the model components from files
def load_files():
    save_dir = 'model_components'
    
    # Load tokenizer
    with open(os.path.join(save_dir, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
        
    # Load label encoder
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
        
    # Load max_length, embedding_dim, embedding_matrix if required
    with open(os.path.join(save_dir, 'max_length.pkl'), 'rb') as f:
        max_length = pickle.load(f)
        
    # Load the trained model
    model = load_model(os.path.join(save_dir, 'best_model.keras'))
    
    return model, tokenizer, label_encoder, max_length

# Prediction function
def classify_text(model, tokenizer, label_encoder, max_length, input_text):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = np.pad(sequences, [(0, 0), (0, max_length - len(sequences[0]))], mode='constant')
    
    # Predict the class
    prediction = model.predict(padded_sequences)
    
    # Convert prediction to label
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_label[0]

# Streamlit UI
def main():
    st.title("Text Classifier")
    
    # Text input
    user_input = st.text_input("Enter text to classify")
    
    # Load model and related components
    model, tokenizer, label_encoder, max_length = load_files()
    
    if st.button("Classify"):
        if user_input:
            # Classify input text
            result = classify_text(model, tokenizer, label_encoder, max_length, user_input)
            st.success(f"Predicted Class: {result}")
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()