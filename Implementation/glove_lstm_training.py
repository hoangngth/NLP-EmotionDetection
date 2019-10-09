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
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten, Dropout, TimeDistributed, Activation
from keras import optimizers
from keras.utils.np_utils import to_categorical   

# Start counting time
start_time = time.time()
print('Training GloVe-LSTM Model...')

utterances = []
labels = []

# LOAD DATA
emotion_dataset_dir = os.getcwd()+'/Dataset/starterkitdata/train.csv'
df = pd.read_csv(emotion_dataset_dir)

utterances = df['Utterances']
labels = df['Label']

labels = np.asarray(labels)

# Word Tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(utterances) # Generate tokens by counting frequency

# Save tokenizer
with open('Tokenizer/tokenizer.30k.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
vocab_size = len(tokenizer.word_index)+1
sequences = tokenizer.texts_to_sequences(utterances) # Turn text into sequence of numbers

max_len = 50 # Make all sequences 50 words long
data = pad_sequences(sequences, maxlen=max_len, padding='post')
print(data.shape) # We have 5509, 100 word sequences now

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
    
# Load word embedding, process WE file
we_glove_dir = os.getcwd() + '/Word_Embedding/glove.twitter.27B.50d.txt'
embedding_index = dict()
print('Converting into dictionary of vectorized words...')
f = open(we_glove_dir, encoding='utf-8', errors='ignore')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()
print('Done.')

# Create a weight matrix for words in training
embedding_dim = 50
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.zeros(embedding_dim, )

# Define Model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = max_len, weights = [embedding_matrix], trainable = False))
model.add(LSTM(32, return_sequences = True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(4))
model.add(Activation('softmax'))
#adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_val, y_val))

scores = model.evaluate(x_test, y_test, verbose=0)
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
y_pred = model.predict(x_test, batch_size=64, verbose=1)
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names= ['Happy', 'Sad', 'Angry', 'Others']))


# Save the trained model
model.save(os.getcwd()+'/Model/Emotional.GloVe.300d-LSTM_model.h5')
print('Model saved to '+os.getcwd()+'/Model/Emotional.GloVe.300d-LSTM_model.h5')

print("Total training time: %s seconds" % (time.time() - start_time))
