# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from kmodes.kmodes import KModes

#%%

# importing cleaned data for cgMLST or wgMLST
MLST_type = "cg" # cg or wg
cleaned_data = pd.read_csv(f"cleaned_data_forML/{MLST_type}MLSTcleaned_data_forML.csv")

#%%

# spliting source labels, MLST-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
MLST_data = cleaned_data.iloc[:, 1:-1]
labels = cleaned_data.Source

#%%

# setting seed for reproducibility because of random centroid start
np.random.seed(3)
# doing k_mode clustering for 5 clusters with random centroid start
km = KModes(n_clusters=5, init='random', verbose=1)
clusters = km.fit_predict(MLST_data)

#%% 

# dataframe of the cluster for each isolat and the source
labels_true = [list(labels)]
clusters_predicted = [list(clusters)]

df_input = labels_true + clusters_predicted  
column_headers = ["Source","Cluster"]

cluster_df = pd.DataFrame(dict(zip(column_headers, df_input)))

# saving dataframe of clusters and source for the isolates
cluster_df.to_csv(f"cluster_{MLST_type}MLSTdf.csv", index=False)
