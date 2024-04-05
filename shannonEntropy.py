# -*- coding: utf-8 -*-


# imports
import pandas as pd 
import numpy as np
from scipy.stats import entropy

#%%

# importing cleaned data for cgMLST or wgMLST
MLST_type = "wg" # cg or wg
cleaned_data = pd.read_csv(f"cleaned_data_forML/{MLST_type}MLSTcleaned_data_forML.csv")

#%%

# spliting source labels, cg/wgMLST-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
MLST_data = cleaned_data.iloc[:, 1:-1]

#%% 

# computing shannon entropy for columns
MLST_data_count = MLST_data.apply(lambda x: x.value_counts())
MLST_data_count.fillna(0, inplace=True)
MLST_data_abundance = MLST_data_count.apply(lambda x: x/(MLST_data_count.shape[0]))
s_entropy = MLST_data_abundance.apply(lambda x: entropy(x))

# calculating max entropy
max_entropy = np.log(MLST_data.shape[0])

# saving shannon entropy 
s_entropy.to_csv(f"shannonEntropy_{MLST_type}MLST_data.csv", index=True)
