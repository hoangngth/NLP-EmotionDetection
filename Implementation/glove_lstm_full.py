import time
import os
import numpy as np
import pandas
import random
import tensorflow as tf
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
texts = []
labels = []

# DATA PRE-PROCESSING
emotion_dataset_dir = os.getcwd()+'/Dataset/4_labels_emo.txt'
print(emotion_dataset_dir,)
with open(emotion_dataset_dir, encoding='utf-8', errors='ignore') as f:
    raw_dataset = f.readlines()
raw_dataset.pop(0)
f.close()

for row in raw_dataset:
    temp_splitted = row.split('\t')
    labels.append(temp_splitted[4].strip('\n'))
    temp_splitted.pop(4)
    temp_splitted.pop(0)
    utterances.append(temp_splitted)
    
# Count the number of each emotion
Counter(labels)

# Convert into csv for visualization
df = pandas.DataFrame(data={"Utterances": utterances, "Label": labels})
df.to_csv("./Dataset/4_emo.csv", sep=',',index=False)

# Randomly take "others" label
new_utterances = []
new_labels = []

random.seed(123)
for i in range(0, len(labels)):
    if labels[i] == 'others':
        random_temp = random.randrange(1,100, 1)
        if (random_temp <= 6):
            new_utterances.append(utterances[i])
            new_labels.append(labels[i])
    else: 
        new_utterances.append(utterances[i])
        new_labels.append(labels[i])
            
Counter(new_labels)

# Convert into csv for visualization
df = pandas.DataFrame(data={"Utterances": new_utterances, "Label": new_labels})
df.to_csv("./Dataset/4_emo_reduced.csv", sep=',',index=False)

for row in new_utterances:
    texts.append(' '.join(row))

# Word Tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) # Generate tokens by counting frequency
vocab_size = len(tokenizer.word_index)+1
sequences = tokenizer.texts_to_sequences(texts) # Turn text into sequence of numbers

max_len = 50 # Make all sequences 50 words long
data = pad_sequences(sequences, maxlen=max_len, padding='post')
print(data.shape) # We have 5509, 100 word sequences now

new_labels = np.asarray(new_labels)
print(new_labels.shape)

# Determine train and validation data
train_valtest_ratio = 0.6 # validation and test set will take 40% of the data
val_test_ratio = 0.5 # the data for validation and test will be split in half, which is 20% of the data

training_samples = round(train_valtest_ratio * data.shape[0])
val_test_samples = data.shape[0] - training_samples
validation_samples = round(val_test_samples * val_test_ratio)
test_samples = data.shape[0] - training_samples - validation_samples

# Split data
x_train = data[:training_samples]
y_train = new_labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = new_labels[training_samples: training_samples + validation_samples]

x_test = data[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = new_labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]

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
#emotional_glove_dir = os.getcwd() + '/Word_Embedding/glove.twitter.27B.50d.txt'
emotional_glove_dir = os.getcwd() + '/Word_Embedding/em-glove.6B.300d-20epoch.txt'
embedding_index = dict()
print('Converting into dictionary of vectorized words...')
f = open(emotional_glove_dir, encoding='utf-8', errors='ignore')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()
print('Done.')

# Create a weight matrix for words in training
embedding_dim = 300
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
model.add(LSTM(48, return_sequences = True))
#model.add(LSTM(64, return_sequences=False, input_shape=(max_len, 3)))
#model.add(LSTM(64, return_sequences=True))
#model.add(LSTM(64, return_sequences=False))
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
                    batch_size=32,
                    validation_data=(x_val, y_val))

scores = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

pyplot.plot(history.history['acc'],label='Training Accuracy')
pyplot.plot(history.history['val_acc'],label='Validation Accuracy')

pyplot.legend()
pyplot.show()

# Save the trained model
model.save(os.getcwd()+'/Model/GloVe.50d-LSTM_model.h5')
print('Model saved to '+os.getcwd()+'/Model/GloVe.50d-LSTM_model.h5')

print("Total training time: %s seconds" % (time.time() - start_time))

