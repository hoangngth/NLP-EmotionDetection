import os
import pickle
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load model for prediction
model = load_model(os.getcwd()+'/Model/Emotional.GloVe.300d-LSTM_model.h5')
max_len = 50

# Load tokenizer
with open('Tokenizer/tokenizer.30k.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
def predict_sentence(input_text):
    seq = tokenizer.texts_to_sequences([input_text])
    print('raw seq:',seq)
    seq = pad_sequences(seq, maxlen= max_len, padding='post')
    print('padded seq:',seq)
    # Predict
    prediction_softmax = model.predict(seq)
    print('Softmax_Happy-Sad-Angry-Others: \n', prediction_softmax)
    return prediction_softmax
    
# Input text 
print('User1 (turn1): ')
input_text = input()
user1_turn1_pred = predict_sentence(input_text)

print('User 2 (turn1): ')
input_text = input()
user2_turn1_pred = predict_sentence(input_text)

print('User1 (turn2): ')
input_text = input()
user1_turn2_pred = predict_sentence(input_text)

concatenated_matrix = np.concatenate((user1_turn1_pred, user2_turn1_pred, user1_turn2_pred), axis = 0)
mean_emotion_pred = concatenated_matrix.mean(0)
print('Mean emotion: \n', mean_emotion_pred)