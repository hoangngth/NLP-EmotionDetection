import pandas as pd
import os
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Lambda

# Load Data
train_dir = os.getcwd() + '/Dataset/Processed_Data.csv'
df = pd.read_csv(train_dir)
utterances = df['Utterances']
labels = df['Label']
labels = np.asarray(labels)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(utterances)
# Save tokenized words
with open('Tokenizer/tokenizer.30k.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(utterances) # Turn text into sequence of numbers
max_len = 100 # Make all sequences 100 words long
data = pad_sequences(sequences, maxlen=max_len, padding='post')
print('Data shape: ', data.shape)

# Determine train and validation data
train_valtest_ratio = 0.7 # validation and test set will take 30% of the data
val_test_ratio = 0.5 # the data for validation and test will be split in half, which is 15% of the data

training_samples = round(train_valtest_ratio * data.shape[0])
val_test_samples = data.shape[0] - training_samples
validation_samples = round(val_test_samples * val_test_ratio)
test_samples = data.shape[0] - training_samples - validation_samples

# Split data
x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

x_test = data[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]

# Convert string label to int
def label_to_number(label):
    switcher = {
                "happy": 0,
                "sad": 1,
                "angry": 2,
                "others": 3,
    }
    return switcher.get(label, "nothing") 
    
for i in range(len(y_train)):
    y_train[i] = label_to_number(y_train[i])
for i in range(len(y_val)):
    y_val[i] = label_to_number(y_val[i])
for i in range(len(y_test)):
    y_test[i] = label_to_number(y_test[i])

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# ELMo is a deep contextualized word vectors
url = 'https://tfhub.dev/google/elmo/2'
elmo = hub.Module(url, trainable=True)


# Input layer
word_input_layer = Input(shape=(max_len, ))
elmo_input_layer = Input(shape=(max_len, ))

# Load word embedding, process WE file
we_dir = '/Word_Embedding/em-glove.6B.300d-20epoch.txt'
emb_dim = 300
embedding_index = dict()
print('[GloVe] Converting into dictionary of vectorized words...')
f = open(we_dir, encoding='utf-8', errors='ignore')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()
print('Done.')
    
# Create a weight matrix for words in training
embedding_matrix = np.zeros((vocab_size, emb_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else: 
        embedding_matrix[i] = np.zeros(emb_dim, )
        
# Output layer
word_output_layer = Embedding(vocab_size, emb_dim, input_length = max_len, weights = [embedding_matrix], trainable = False)(word_input_layer)
elmo_output_layer = Lambda(elmo.to_keras_layers)
model.add(Bidirectional(LSTM(32, return_sequences = True), input_shape=(32, 3, max_len)))
model.add(Bidirectional(LSTM(32)))
return model
# Build model

print('Done.')
