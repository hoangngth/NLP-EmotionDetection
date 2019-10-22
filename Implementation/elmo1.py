# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:12:10 2019

@author: USER
"""

import tensorflow_hub as hub
import tensorflow as tf

print('Processing...')
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
print('done')
print('Processing...')
embeddings = elmo(
    ["the cat is on the mat", "dogs are in the fog"],
    signature="default",
    as_dict=True)["elmo"]