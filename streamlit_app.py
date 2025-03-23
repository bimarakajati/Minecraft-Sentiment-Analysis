from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import streamlit as st

# Set the web favicon and title
st.set_page_config(page_title="Minecraft Sentiment Analysis", page_icon="⛏️")

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('Model/bilstm_model.keras')
    with open('Model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

loaded_bilstm_model, loaded_tokenizer = load_model_and_tokenizer()

MAX_SEQUENCE_LENGTH = 250
labels = ['Negative', 'Neutral', 'Positive']

st.title('⛏️ Minecraft Sentiment Analysis')
st.write("**Source:** [Minecraft Reviews](https://play.google.com/store/apps/details?id=com.mojang.minecraftpe&hl=en&gl=US)")

# User input
user_input = st.text_area("Enter your review here:", height=150)

if st.button('Analyze Sentiment'):
    if user_input:
        # Preprocess and predict
        test_seq = loaded_tokenizer.texts_to_sequences([user_input])
        test_padded = pad_sequences(test_seq, maxlen=MAX_SEQUENCE_LENGTH)
        prediction = loaded_bilstm_model.predict(test_padded)

        # Get the predicted label and probabilities
        predicted_label = labels[np.argmax(prediction)]
        probabilities = prediction[0]

        # Display results
        st.write(f"Predicted Sentiment: **{predicted_label}**")

        # Visualize probabilities using a bar chart
        st.bar_chart(dict(zip(labels, probabilities)))
    else:
        st.write("Please enter a review to analyze.")

st.write("---")
st.write("This app uses a BiLSTM model to predict the sentiment of your review.")