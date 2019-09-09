import os
from collections import Counter
import pandas
import random

utterances = []
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
            
print(Counter(new_labels))

# Convert into csv for visualization and training
df = pandas.DataFrame(data={"Utterances": new_utterances, "Label": new_labels})
df.to_csv("./Dataset/4_emo_reduced.csv", sep=',',index=False)