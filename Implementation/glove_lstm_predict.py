import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

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

model = load_model(os.getcwd()+'/Model/GloVe.50d-LSTM_model.h5')

my_text = 'wut'
seq = tokenizer.texts_to_sequences([my_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen= max_len, padding='post')
print('padded seq:',seq)
prediction_softmax = model.predict(seq)
print('Softmax: ', prediction_softmax)