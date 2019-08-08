import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

utterances = []
texts = []
labels = []

# DATA PRE-PROCESSING
with open(os.getcwd()+'/Dataset/4_labels_emo.txt', encoding='utf-8', errors='ignore') as f:
    raw_dataset = f.readlines()
raw_dataset.pop(0)

for row in raw_dataset:
    temp_splitted = row.split('\t')
    labels.append(temp_splitted[4])
    temp_splitted.pop(4)
    temp_splitted.pop(0)
    utterances.append(temp_splitted)
    
for row in utterances:
    texts.append(' '.join(row))

# Word Tokenizing
max_words = 10000 # We will only consider the 10K most used words in this dataset
tokenizer = Tokenizer(num_words=max_words) # Setup
tokenizer.fit_on_texts(texts) # Generate tokens by counting frequency
sequences = tokenizer.texts_to_sequences(texts) # Turn text into sequence of numbers

maxlen = 100 # Make all sequences 100 words long
data = pad_sequences(sequences, maxlen=maxlen)
print(data.shape) # We have 25K, 100 word sequences now
