import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

st.title('Next Word Prediction --- Pride and Prejudice by Jane Austen')

st.subheader('a) Model Summary')
st.write('In this LSTM model, a tokenizer is initialized and trained on the text, establishing a mapping of words to unique integer indices. Input sequences are generated using n-grams, which consist of sequences of words for training. Each line of text is tokenized, and sequences are created for each token. The maximum sequence length is determined, and all input sequences are padded to maintain a uniform size.')
st.write('The variable X contains the padded input sequences, while y holds the one-hot encoded labels for the next word. A Sequential model is built with an embedding layer, three LSTM layers, batch normalization, and a final dense layer with a softmax activation for multi-class predictions. The model is compiled using categorical crossentropy loss and optimized with the Adam algorithm.')
st.write("During training, the model is trained for 50 epochs, monitoring both training and validation accuracy. During training, the validation accuracy surpasses 80%. After completing the training, the model's performance is logged in a CSV file, and both the trained model and tokenizer are saved to disk for future reference.")

model_df = pd.read_csv(r'C:\Users\Edwin\Python\bootcamp\Projects\lstm2\model_performance.csv')
model_df['epochs'] = model_df.index + 1

st.line_chart(model_df, x = 'epochs', y = ['Train Accuracy', 'Test Accuracy'],
              x_label = 'Epoch(s)', y_label = 'Accuracy', color=["#FF0000", "#0000FF"])


st.subheader('b) Next Word Prediction App')
st.write("Enter the text you want and select the number of words to predict. Enjoy! Feel free to check out the source code on GitHub.")
# Generate next word predictions
seed_text = st.text_input("Enter text here:")
next_words = st.number_input("Number of words to predict:", min_value = 1, max_value = 30, value = 5)

with open(r'C:\Users\Edwin\Python\bootcamp\Projects\lstm2\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Open model
lstm_model = load_model(r"C:\Users\Edwin\Python\bootcamp\Projects\lstm2\next_word_predictor.h5")

if st.button('Predict'):
    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen = 24, padding='pre')
        predicted_probs = lstm_model.predict(token_list, verbose = 0)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        generated_text += " " + predicted_word

    st.write("Prediction: ", generated_text)