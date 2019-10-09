# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:02:17 2019

@author: Nguyen The Hoang
"""

import os
from gensim.models.keyedvectors import KeyedVectors

print('Converting...')
model = KeyedVectors.load_word2vec_format(os.getcwd() + '/GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format(os.getcwd()+'/GoogleNews-vectors-negative300.txt', binary=False)
print('Done.')