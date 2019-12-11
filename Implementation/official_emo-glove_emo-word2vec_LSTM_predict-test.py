import os
import pandas as pd
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load model for prediction
model = load_model(os.getcwd()+'/Model/official_emglove-emw2v_model.h5')
emotion_dataset_test_dir = os.getcwd()+'/Dataset/conversations_test_preprocessed-spellcorrection-demoji.csv'
max_len = 50

# Load tokenizer, test file
with open('Tokenizer/spellcorrection-demoji_tokenizer.30k.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
df_test = pd.read_csv(emotion_dataset_test_dir)   
    
def predict_sentence(input_text):
    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen= max_len, padding='post')
    # Predict
    prediction_softmax = model.predict(seq)
    return prediction_softmax

predict_results = []
# Input text 
for index, row in df_test.iterrows():
    predict_result = predict_sentence(row['Utterances'])
    predict_results.append(predict_result)
    print("Predicted: " + str(index))

predict_results_matrix = np.concatenate(predict_results, axis=0)
df_new = pd.concat([df_test, pd.DataFrame(predict_results_matrix)], axis=1)
df_new.to_csv(os.getcwd()+'/Result/official_emglove-emw2v_predicted-conversations.csv')