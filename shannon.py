# -*- coding: utf-8 -*-


# imports
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy


#%%

# importing cleaned data for cgMLST or wgMLST
MLST_type = "wg" # cg or wg
cleaned_data = pd.read_csv(f"cleaned_data_forML/{MLST_type}MLSTcleaned_data_forML.csv")

#%%

# spliting source labels, cg/wgMLST-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
MLST_data = cleaned_data.iloc[:, 1:-1]
labels = cleaned_data.Source
sample_id = cleaned_data.SRA_no

# making a datafram with cg/wgMLST and labels
MLST_labeledData =  cleaned_data.iloc[:, 1:]

# encode lables as integers
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# saving label integer and source name in dictionary
label_dict = dict(zip((encoder.transform(encoder.classes_)), encoder.classes_ ))

#%% 

# computing shannon entropy for columns
MLST_data_count = MLST_data.apply(lambda x: x.value_counts())
MLST_data_count.fillna(0, inplace=True)
MLST_data_abundance = MLST_data_count.apply(lambda x: x/(MLST_data_count.shape[0]))
s_entropy = MLST_data_abundance.apply(lambda x: entropy(x))

# calculating normalized entropy
max_entropy = np.log(MLST_data.shape[0])
norm_entropy = s_entropy.div(max_entropy)

# saving shannon entropy 
s_entropy.to_csv(f"shannonEntropy_{MLST_type}MLST_data.csv", index=True)
norm_entropy.to_csv(f"shannonNormal_{MLST_type}MLST_data.csv", index=True)

#%%

# computing shannon entropy for columns per source

MLST_slugs = MLST_labeledData[MLST_labeledData["Source"]=="slugs"]
MLST_dairyfarm = MLST_labeledData[MLST_labeledData["Source"]=="dairy farm"]
MLST_environment = MLST_labeledData[MLST_labeledData["Source"]=="rural/urban"]
MLST_salmon = MLST_labeledData[MLST_labeledData["Source"]=="salmon"]
MLST_meat = MLST_labeledData[MLST_labeledData["Source"]=="meat"]


# computing shannon entropy for columns in slugs
MLST_slugs_count = MLST_slugs.iloc[:, :-1].apply(lambda x: x.value_counts())
MLST_slugs_count.fillna(0, inplace=True)
MLST_slugs_abundance = MLST_slugs_count.apply(lambda x: x/(MLST_slugs_count.shape[0]))
s_entropy_slugs = MLST_slugs_abundance.apply(lambda x: entropy(x))
# calculating normalized entropy
max_entropy_slugs = np.log(MLST_slugs.shape[0])
norm_entropy_slugs = s_entropy_slugs.div(max_entropy_slugs)

# computing shannon entropy for columns in dairy farm
MLST_dairyfarm_count = MLST_dairyfarm.iloc[:, :-1].apply(lambda x: x.value_counts())
MLST_dairyfarm_count.fillna(0, inplace=True)
MLST_dairyfarm_abundance = MLST_dairyfarm_count.apply(lambda x: x/(MLST_dairyfarm_count.shape[0]))
s_entropy_dairyfarm = MLST_dairyfarm_abundance.apply(lambda x: entropy(x))
# calculating normalized entropy
max_entropy_dairyfarm= np.log(MLST_dairyfarm.shape[0])
norm_entropy_dairyfarm = s_entropy_dairyfarm.div(max_entropy_dairyfarm)

# computing shannon entropy for columns in environment
MLST_environment_count = MLST_environment.iloc[:, :-1].apply(lambda x: x.value_counts())
MLST_environment_count.fillna(0, inplace=True)
MLST_environment_abundance = MLST_environment_count.apply(lambda x: x/(MLST_environment_count.shape[0]))
s_entropy_environment = MLST_environment_abundance.apply(lambda x: entropy(x))
# calculating normalized entropy
max_entropy_environment = np.log(MLST_environment.shape[0])
norm_entropy_environment = s_entropy_environment.div(max_entropy_environment)

# computing shannon entropy for columns in salmon
MLST_salmon_count = MLST_salmon.iloc[:, :-1].apply(lambda x: x.value_counts())
MLST_salmon_count.fillna(0, inplace=True)
MLST_salmon_abundance = MLST_salmon_count.apply(lambda x: x/(MLST_salmon_count.shape[0]))
s_entropy_salmon = MLST_salmon_abundance.apply(lambda x: entropy(x))
# calculating normalized entropy
max_entropy_salmon = np.log(MLST_salmon.shape[0])
norm_entropy_salmon = s_entropy_salmon.div(max_entropy_salmon)

# computing shannon entropy for columns in meat
MLST_meat_count = MLST_meat.iloc[:, :-1].apply(lambda x: x.value_counts())
MLST_meat_count.fillna(0, inplace=True)
MLST_meat_abundance = MLST_meat_count.apply(lambda x: x/(MLST_meat_count.shape[0]))
s_entropy_meat = MLST_meat_abundance.apply(lambda x: entropy(x))
# calculating normalized entropy
max_entropy_meat = np.log(MLST_meat.shape[0])
norm_entropy_meat = s_entropy_meat.div(max_entropy_meat)

# saving shannon entropys 
s_entropy_slugs.to_csv(f"shannonEntropy_{MLST_type}MLST_slugs.csv", index=True)
s_entropy_dairyfarm.to_csv(f"shannonEntropy_{MLST_type}MLST_dairyfarm.csv", index=True)
s_entropy_environment.to_csv(f"shannonEntropy_{MLST_type}MLST_environment.csv", index=True)
s_entropy_salmon.to_csv(f"shannonEntropy_{MLST_type}MLST_salmon.csv", index=True)
s_entropy_meat.to_csv(f"shannonEntropy_{MLST_type}MLST_meat.csv", index=True)


norm_entropy_slugs.to_csv(f"shannonNormal_{MLST_type}MLST_slugs.csv", index=True)
norm_entropy_dairyfarm.to_csv(f"shannonNormal_{MLST_type}MLST_dairyfarm.csv", index=True)
norm_entropy_environment.to_csv(f"shannonNormal_{MLST_type}MLST_environment.csv", index=True)
norm_entropy_salmon.to_csv(f"shannonNormal_{MLST_type}MLST_salmon.csv", index=True)
norm_entropy_meat.to_csv(f"shannonNormal_{MLST_type}MLST_meat.csv", index=True)
