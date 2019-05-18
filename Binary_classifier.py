# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import binary_classifier_rnn_word2vec as binary_rnn
import preprocess as pp
train_data=pd.read_csv('training_data_original.csv')
train_data=train_data.drop(['ProjectID'], axis=1)
print('Number of words before data preprocessing: ',train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())
train_data=pp.preprocess(train_data)
print('Number of words after data preprocessing: ',train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())
train_data=pp.lemtext(train_data)
print('Number of words after lemmatization: ',train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())
train_data=pp.numbertoword(train_data) 
print('Number of words after number conversion: ',train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())


binary_rnn.binary_classifier(train_data)

