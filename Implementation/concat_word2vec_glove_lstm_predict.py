import os
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load model for prediction
model = load_model(os.getcwd()+'/Model/Final_model.h5')
max_len = 50

# Input text 
print('Enter sentence: ')
input_text = input()

# Load tokenizer
with open('Tokenizer/tokenizer.30k.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

seq = tokenizer.texts_to_sequences([input_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen= max_len, padding='post')
print('padded seq:',seq)

# Predict
prediction_softmax = model.predict(seq)
print('Softmax_Happy-Sad-Angry-Others: \n', prediction_softmax)