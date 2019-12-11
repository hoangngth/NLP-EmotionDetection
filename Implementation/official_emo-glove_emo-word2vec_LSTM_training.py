import time
import csv
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Flatten, Dropout, Activation, Concatenate, Input, Add, LeakyReLU, Bidirectional
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate

# Start counting time
start_time = time.time()

utterances = []
utterances_test = []
utterances_all = []
labels = []
labels_test = []

# LOAD DATA
emotion_dataset_dir = os.getcwd()+'/Dataset/preprocessed-spellcorrection-demoji_train.csv'
emotion_dataset_test_dir = os.getcwd()+'/Dataset/preprocessed-spellcorrection-demoji_test.csv'
df = pd.read_csv(emotion_dataset_dir)
df_test = pd.read_csv(emotion_dataset_test_dir)

utterances = df['Utterances']
utterances_test = df_test['Utterances']
utterances_all = pd.concat([utterances, utterances_test])
labels = df['Label']
labels_test = df_test['Label']

labels = np.asarray(labels)
labels_test = np.asarray(labels_test)

# Word Tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(utterances_all) # Generate tokens by counting frequency

# Save tokenizer
with open('Tokenizer/spellcorrection-demoji_tokenizer.30k.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocab_size = len(tokenizer.word_index)+1
sequences = tokenizer.texts_to_sequences(utterances) # Turn text into sequence of numbers
sequences_test = tokenizer.texts_to_sequences(utterances_test)

max_len = 50 # Make all sequences 50 words long
data = pad_sequences(sequences, maxlen=max_len, padding='post')
data_test = pad_sequences(sequences_test, maxlen=max_len, padding='post')
print(data.shape) # We have 5509, 100 word sequences now
print(data_test.shape)

# Determine train and validation data
train_val_ratio = 0.9

training_samples = round(train_val_ratio * data.shape[0])
validation_samples = data.shape[0] - training_samples

# Determine test data
test_samples = data_test.shape[0]

# Split data
x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

x_test = data_test[:test_samples]
y_test = labels_test[:test_samples]


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

#----------------------------------------------------------
# MODEL 1 (GloVe-LSTM)
def train_glove_lstm(we_dir, emb_dim):
    # Load word embedding, process WE file
    embedding_index = dict()
    print('Converting into dictionary of vectorized words...')
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
            
    # Define Model
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim, input_length = max_len, weights = [embedding_matrix], trainable = False))
    model.add(Bidirectional(LSTM(64, return_sequences = True)))

    return model
#----------------------------------------------------------

#----------------------------------------------------------
# MODEL 2 (Word2Vec-LSTM)
# Load word embedding, process WE file
def train_w2v_lstm(we_dir, emb_dim):
    # Load word embedding, process WE file
    embedding_index = dict()
    print('Converting into dictionary of vectorized words...')
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
            
    # Define Model
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim, input_length = max_len, weights = [embedding_matrix], trainable = False))
    model.add(Bidirectional(LSTM(64, return_sequences = True)))

    return model
#----------------------------------------------------------
    

# We need to Concatenate 2 models together
we_glove_dir = os.getcwd() + '/Word_Embedding/em-glove.6B.300d-20epoch.txt'
model_1 = train_glove_lstm(we_glove_dir, 300)
we_w2v_dir = os.getcwd() + '/Word_Embedding/em-w2v-out-wiki-20epoch.txt'
model_2 = train_w2v_lstm(we_w2v_dir, 300)

# Create placeholder model for concatenation
concatenated_input = Input(shape = (max_len,))
out1 = model_1(concatenated_input)    
out2 = model_2(concatenated_input)   
concatenated_output = Concatenate()([out1,out2])
concatenated_output = LeakyReLU(alpha=.3)(concatenated_output)
concatenated_output = Flatten()(concatenated_output)
concatenated_output = Dense(64)(concatenated_output)
concatenated_output = Dropout(.5)(concatenated_output)
concatenated_output = Dense(4, activation='softmax')(concatenated_output)

final_model = Model(concatenated_input, concatenated_output)
final_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = final_model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_data=(x_val, y_val))
scores = final_model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

pyplot.plot(history.history['acc'],label='Training Accuracy')
pyplot.plot(history.history['val_acc'],label='Validation Accuracy')
pyplot.legend()
pyplot.show()

pyplot.plot(history.history['loss'],label='Training Loss')
pyplot.plot(history.history['val_loss'],label='Validation Loss')
pyplot.legend()
pyplot.show()

# Calculate Precision, Recall, F1
#--- If trained then load model, else comment it out
#from keras.models import load_model
#final_model = load_model(os.getcwd()+'/Model/Final_model.h5')
#---------------------------------------------------
y_pred = final_model.predict(x_test, batch_size=64, verbose=1)
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names= ['Happy', 'Sad', 'Angry', 'Others']))

# Save the trained model
final_model.save(os.getcwd()+'/Model/official_emglove-emw2v_model.h5')
print('Model saved to '+os.getcwd()+'/Model/official_emglove-emw2v_model.h5')

print("Total training time: %s seconds" % (time.time() - start_time))