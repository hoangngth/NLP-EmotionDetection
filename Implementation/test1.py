import os
import numpy as np
import pandas
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten, Dropout, TimeDistributed, Activation
from keras import optimizers

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
    
for row in utterances:
    texts.append(' '.join(row))
    
Counter(labels)

#df = pandas.DataFrame(data={"col1": utterances, "col2": labels})
#df.to_csv("./4_emo.csv", sep=',',index=False)

# Word Tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) # Generate tokens by counting frequency
vocab_size = len(tokenizer.word_index)+1
sequences = tokenizer.texts_to_sequences(texts) # Turn text into sequence of numbers

max_len = 50 # Make all sequences 50 words long
data = pad_sequences(sequences, maxlen=max_len, padding='post')
print(data.shape) # We have 5509, 100 word sequences now

labels = np.asarray(labels)
print(labels.shape)

# Determine train and validation data
training_samples_ratio = 0.8
training_samples = round(training_samples_ratio * data.shape[0])
validation_samples = data.shape[0] - training_samples

# Split data
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

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
    
# Load word embedding, process WE file
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
model.add(LSTM(64, return_sequences=True, input_shape=(max_len, 3)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
