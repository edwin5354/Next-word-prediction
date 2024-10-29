# Import relevant libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, BatchNormalization
from sklearn.model_selection import train_test_split
import pandas as pd

TF_ENABLE_ONEDNN_OPTS=0

# Open file
txt_path =  './LSTM DATA.txt'

with open(txt_path, "r", encoding='utf-8') as f:
    text = f.read()
    f.close()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Assign length of word index
total_words = len(tokenizer.word_index) + 1 # 7561
tokenizer.word_index

# Declare ngrams
input_sequences = []
# Split the sentence 
for line in text.split('\n'):
    #get tokens
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Trial example
sentence_token = input_sequences[7] # [42, 1029]
sentence = []
for token in sentence_token:
    sentence.append(list((tokenizer.word_index).keys())[list((tokenizer.word_index).values()).index(token)])
# sentence ['this', 'ebook']

# maximum sentence length
max_sequence_len = max([len(seq) for seq in input_sequences])

# input sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words)) # One-hot-encoding

# Split into training and validation sets  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Embedding(input_dim = total_words, output_dim = 100, input_length = max_sequence_len - 1))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(BatchNormalization())
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=50,  validation_data = (X_val, y_val), verbose=1)

acc_df = pd.DataFrame({
    'Train Accuracy': history.history['accuracy'],
    'Test Accuracy': history.history.get('val_accuracy', [])
})

model.save('./next_word_predictor.h5')
acc_df.to_csv('./model_performance.csv', index=False)

import pickle
with open('./tokenizer.pickle', 'wb') as handle: 
    pickle.dump(tokenizer, handle, protocol= pickle.HIGHEST_PROTOCOL)