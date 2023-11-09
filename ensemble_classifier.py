# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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
label_dict = dict(zip(range(len(encoder.classes_)), encoder.classes_))

#%%

# split randomly into training(70%) and testing(30%)
cgMLST_train, cgMLST_test, labels_train, labels_test = train_test_split(
        cgMLST_data,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=2)

#%% 

# feature reduction
# random forest feature importance
# sklearn.feature_selection.VarianceThreshold

#%%

# setup for random forest model
model = RandomForestClassifier(random_state=2)

# parameters
param_grid   = [{'n_estimators': [100, 300, 500], 'criterion': ['gini']}]

# gridsearch for best parameters 6-fold cross validation
gs = GridSearchCV(estimator=model, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=6,
                  n_jobs=-1)

#%%

# fiting model and finding best parameters 
gs = gs.fit(cgMLST_train, labels_train)

print(gs.best_params_)
print(gs.best_score_)

# best model
clf = gs.best_estimator_

#%% 

# predicting
labelno_predict = clf.predict(cgMLST_test)
proba_predict = clf.predict_proba(cgMLST_test)
source_predict=[label_dict[x] for x in labelno_predict]

#%% 

# performance metrics
performance_report = classification_report(
            labels_test,
            labelno_predict,
            target_names=label_dict.values())

print(performance_report)

conf_matrix = confusion_matrix(
            labels_test,
            labelno_predict)

ConfusionMatrixDisplay.from_predictions(
            labels_test,
            labelno_predict)

print(label_dict)
print(conf_matrix)

#%%

# dataframe for the probabilityes predicted
predictions = [list(source_predict)]
proba_predict = list(proba_predict.T)
predictions += [list(x) for x in proba_predict]
column_headers = ["prediction"]
column_headers += ["probability_{}".format(label_dict[x])for x in range(len(label_dict.keys()))]

probability_df = pd.DataFrame(dict(zip(column_headers, predictions))).round(decimals=3)
