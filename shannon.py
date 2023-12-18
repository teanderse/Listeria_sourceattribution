# -*- coding: utf-8 -*-


# imports
import pandas as pd 
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

# making a datafram with cgMLST and labels
cgMLST_labeledData =  cleaned_data.iloc[:, 1:]


# encode lables as integers
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# saving label integer and source name in dictionary
label_dict = dict(zip((encoder.transform(encoder.classes_)), encoder.classes_ ))

#%% 

# computing shannon entropy for columns
cgMLST_data_count = cgMLST_data.apply(lambda x: x.value_counts())
cgMLST_data_count.fillna(0, inplace=True)
cgMLST_data_abundance = cgMLST_data_count.apply(lambda x: x/(cgMLST_data_count.shape[0]))
s_entropy = cgMLST_data_abundance.apply(lambda x: entropy(x))


# saving shannon entropy 
# s_entropy.to_csv("shannonEntropy_cgMLST_data.csv", index=True)

#%%

# computing shannon entropy for columns per source

cgMLST_slugs = cgMLST_labeledData[cgMLST_labeledData["Source"]=="slugs"]
cgMLST_dairyfarm = cgMLST_labeledData[cgMLST_labeledData["Source"]=="dairy farm"]
cgMLST_environment = cgMLST_labeledData[cgMLST_labeledData["Source"]=="rural/urban"]
cgMLST_salmon = cgMLST_labeledData[cgMLST_labeledData["Source"]=="salmon"]
cgMLST_meat = cgMLST_labeledData[cgMLST_labeledData["Source"]=="meat"]


# computing shannon entropy for columns in slugs
cgMLST_slugs_count = cgMLST_slugs.iloc[:, :-1].apply(lambda x: x.value_counts())
cgMLST_slugs_count.fillna(0, inplace=True)
cgMLST_slugs_abundance = cgMLST_slugs_count.apply(lambda x: x/(cgMLST_slugs_count.shape[0]))
s_entropy_slugs = cgMLST_slugs_abundance.apply(lambda x: entropy(x))

# computing shannon entropy for columns in dairy farm
cgMLST_dairyfarm_count = cgMLST_dairyfarm.iloc[:, :-1].apply(lambda x: x.value_counts())
cgMLST_dairyfarm_count.fillna(0, inplace=True)
cgMLST_dairyfarm_abundance = cgMLST_dairyfarm_count.apply(lambda x: x/(cgMLST_dairyfarm_count.shape[0]))
s_entropy_dairyfarm = cgMLST_dairyfarm_abundance.apply(lambda x: entropy(x))

# computing shannon entropy for columns in environment
cgMLST_environment_count = cgMLST_environment.iloc[:, :-1].apply(lambda x: x.value_counts())
cgMLST_environment_count.fillna(0, inplace=True)
cgMLST_environment_abundance = cgMLST_environment_count.apply(lambda x: x/(cgMLST_environment_count.shape[0]))
s_entropy_environment = cgMLST_environment_abundance.apply(lambda x: entropy(x))

# computing shannon entropy for columns in salmon
cgMLST_salmon_count = cgMLST_salmon.iloc[:, :-1].apply(lambda x: x.value_counts())
cgMLST_salmon_count.fillna(0, inplace=True)
cgMLST_salmon_abundance = cgMLST_salmon_count.apply(lambda x: x/(cgMLST_salmon_count.shape[0]))
s_entropy_salmon = cgMLST_salmon_abundance.apply(lambda x: entropy(x))

# computing shannon entropy for columns in meat
cgMLST_meat_count = cgMLST_meat.iloc[:, :-1].apply(lambda x: x.value_counts())
cgMLST_meat_count.fillna(0, inplace=True)
cgMLST_meat_abundance = cgMLST_meat_count.apply(lambda x: x/(cgMLST_meat_count.shape[0]))
s_entropy_meat = cgMLST_meat_abundance.apply(lambda x: entropy(x))


# saving shannon entropys 
#s_entropy_slugs.to_csv("shannonEntropy_slugs.csv", index=True)
#s_entropy_dairyfarm.to_csv("shannonEntropy_dairyfarm.csv", index=True)
#s_entropy_environment.to_csv("shannonEntropy_environment.csv", index=True)
#s_entropy_salmon.to_csv("shannonEntropy_salmon.csv", index=True)
#s_entropy_meat.to_csv("shannonEntropy_meat.csv", index=True)