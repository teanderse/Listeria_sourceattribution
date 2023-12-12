# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy

#%%

# importing cleaned data
cleaned_data = pd.read_csv("cleaned_data_forML.csv")

#%%

# spliting source labels, cgmlst-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
cgMLST_data = cleaned_data.iloc[:, 1:-1]
labels = cleaned_data.Source
sample_id = cleaned_data.SRA_no

# encode lables as integers
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# saving label integer and source name in dictionary
label_dict = dict(zip((encoder.transform(encoder.classes_)), encoder.classes_ ))

#%% 

# computing shannon entropy for columns in train
cgMLST_data_count = cgMLST_data.apply(lambda x: x.value_counts())
cgMLST_data_count.fillna(0, inplace=True)
cgMLST_data_abundance = cgMLST_data_count.apply(lambda x: x/(cgMLST_data_count.shape[0]))
s_entropy = cgMLST_data_abundance.apply(lambda x: entropy(x))


# saving shannon entropy 
# s_entropy.to_csv("shannonEntropy_cgMLST_data.csv", index=True)
