# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np 
from kmodes.kmodes import KModes

#%%

# importing cleaned data
cleaned_data = pd.read_csv("cleaned_data_forML.csv")

#%%

# spliting source labels, cgmlst-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
cgMLST_data = cleaned_data.iloc[:, 1:-1]
labels = cleaned_data.Source
sample_id = cleaned_data.SRA_no

#%%

np.random.seed(3)
# doing k_mode clustering for 5 clusters
km = KModes(n_clusters=5, init='random', verbose=1)

clusters = km.fit_predict(cgMLST_data)

#%% 

# dataframe for the clusters predicted and their source

labels_true = [list(labels)]
clusters_predicted = [list(clusters)]

df_input = labels_true + clusters_predicted  
column_headers = ["Source","Cluster"]

cluster_df = pd.DataFrame(dict(zip(column_headers, df_input)))

# saving performance result test data
cluster_df.to_csv("cluster_df.csv", index=False)

