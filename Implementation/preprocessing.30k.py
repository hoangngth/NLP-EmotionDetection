import os
from collections import Counter
import pandas

utterances = []
labels = []

# DATA PRE-PROCESSING
emotion_dataset_dir = os.getcwd()+'/Dataset/starterkitdata/train.txt'
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
df.to_csv("./Dataset/starterkitdata/train.csv", sep=',',index=False)