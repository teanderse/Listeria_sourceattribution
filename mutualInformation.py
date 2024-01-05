# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%%
# Testing effects of mutual information on controled test dataframe
# column with same values as classes in y
col1 = [1, 2, 3, 4, 5]*200
# column with one value fore each class in y, but with som values being large
col2 = [5, 50, 7, 700, 400]*200
# column with only one value
col3 = [1, 1, 1, 1, 1]*200
# column y holding classes
coly = [1, 2, 3, 4, 5]*200

# making datafram of columns
test_df = pd.DataFrame({'col1': col1,'col2': col2,'col3': col3,'coly': coly})

yencoder = LabelEncoder()
scaler = StandardScaler()
xencoder = OrdinalEncoder()

# y being label encoded as
y = np.array(test_df.coly)
y = yencoder.fit_transform(y)

# x, x scaled and x label encoded
x = test_df.iloc[:, 0:-1]
x_scaled = scaler.fit_transform(x)
x_encoded = xencoder.fit_transform(x)

#%% 

# computing mutual information for columns in x with classes in y
mutualI_test_raw = mutual_info_classif(x, y, random_state=3, discrete_features=True)
mutualI_test_raw = pd.DataFrame({'MI_raw':mutualI_test_raw})
mutualI_test_raw.index = x.columns
raw = mutualI_test_raw.sort_values(by="MI_raw", ascending=False)

# computing mutual information for columns in scaled x with classes in y
mutualI_test_scaled = mutual_info_classif(x_scaled, y, random_state=3)
mutualI_test_scaled = pd.DataFrame({'MI_scaled':mutualI_test_scaled})
mutualI_test_scaled.index = x.columns
scaled = mutualI_test_scaled.sort_values(by="MI_scaled", ascending=False)

# computing mutual information for columns in label encoded x with classes in y
mutualI_test_encoded = mutual_info_classif(x_encoded, y, random_state=3, discrete_features=True)
mutualI_test_encoded = pd.DataFrame({'MI_encoded':mutualI_test_encoded})
mutualI_test_encoded.index = x.columns
encoded = mutualI_test_encoded.sort_values(by="MI_encoded", ascending=False)

mutualInformation_test_dis = raw.merge(scaled,left_index=True,right_index=True)
mutualInformation_test_dis = mutualInformation_test_dis.merge(encoded,left_index=True,right_index=True)

 
#%%
# ---------------------------------------------------

# Testing effect of label encoding training data 

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

#%%

# split randomly into training(70%) and testing(30%)
cgMLST_train, cgMLST_test, labels_train, labels_test = train_test_split(
        cgMLST_data,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=3)

#%% 

# computing mutual information for columns in cgMLST_train
mutualI_raw = mutual_info_classif(cgMLST_train, labels_train, random_state=3, discrete_features=True)
mutualI_raw = pd.DataFrame({'MI_raw':mutualI_raw})
mutualI_raw.index = cgMLST_train.columns
raw_cgMLST = mutualI_raw.sort_values(by="MI_raw", ascending=False)
raw_cgMLST.head()

# encoding features before feature selection
feature_encoder = OrdinalEncoder()
cgMLST_train_encoded = feature_encoder.fit_transform(cgMLST_train)

# computing mutual information for columns in scaled cgMLST_train
mutualI_encoded = mutual_info_classif(cgMLST_train_encoded, labels_train, random_state=3, discrete_features=True)
mutualI_encoded = pd.DataFrame({'MI_encoded':mutualI_encoded})
mutualI_encoded.index = cgMLST_train.columns
encoded_cgMLST = mutualI_encoded.sort_values(by="MI_encoded", ascending=False)
encoded_cgMLST.head()

# Comparing raw cgMLST trainigdata with the scaled cgMLST training data
mutualInformation_cgMLST = raw_cgMLST.merge(encoded_cgMLST,left_index=True,right_index=True)
mutualInformation_cgMLST["divergens"] = mutualInformation_cgMLST["MI_raw"]-mutualInformation_cgMLST["MI_encoded"]

print("Max:",mutualInformation_cgMLST["divergens"].max(), "   Min:",mutualInformation_cgMLST["divergens"].min())
# Max: 0    Min: 0

# saving mutual information calculation for features in train witout scale
#raw_cgMLST.to_csv("mutualInfo_trainingdata_discrete.csv", index=True)
