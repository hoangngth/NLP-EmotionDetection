import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# LOAD DATA
emotion_dataset_dir = os.getcwd()+'/Dataset/4_emo_reduced.csv'
df = pd.read_csv(emotion_dataset_dir)
utterances = df['Utterances']

# Load model for prediction
model = load_model(os.getcwd()+'/Model/GloVe.50d-LSTM_model.h5')
max_len = 50

# Input text 
print('Enter sentence: ')
input_text = input()

# Tokenizing input text for prediction
tokenizer = Tokenizer()
tokenizer.fit_on_texts(utterances) # Generate tokens by counting frequency
seq = tokenizer.texts_to_sequences([input_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen= max_len, padding='post')
print('padded seq:',seq)

# Predict
prediction_softmax = model.predict(seq)
print('Softmax_Happy-Sad-Angry-Others: \n', prediction_softmax)