import os

utterances = []
labels = []

# DATA PRE-PROCESSING
with open(os.getcwd()+'/Dataset/4_labels_emo.txt', encoding='utf-8', errors='ignore') as f:
    raw_dataset = f.readlines()

for row in raw_dataset:
    temp_splitted = row.split('\t')
    labels.append(temp_splitted[4])
    temp_splitted.pop(4)
    temp_splitted.pop(0)
    utterances.append(temp_splitted)


    
    


