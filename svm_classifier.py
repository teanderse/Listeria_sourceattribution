# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:47:17 2023

@author: tessa
"""

# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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
label_dict = dict(zip((encoder.transform(encoder.classes_)), encoder.classes_ ))

#%%

# split randomly into training(70%) and testing(30%)
cgMLST_train, cgMLST_test, labels_train, labels_test = train_test_split(
        cgMLST_data,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=3)

#%% 

# parameter C
param_range  = [0.01, 0.03, 0.05, 0.07,  0.1, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
# scaling parameter gamma for rbf-kernel
param_range2 = [0.0001, 0.001,0.005, 0.01, 0.015, 0.1, 1.0, 5.0]   
   
param_grid   = [{'C': param_range, 'gamma': param_range2, 'kernel': ['rbf']}]


# setup for support vector clasifier 
SVM_model = SVC(random_state=2)


gs = GridSearchCV(estimator=SVM_model, 
                  param_grid=param_grid, 
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}), 
                  cv=5,
                  refit='weighted_f1',
                  return_train_score=True,
                  n_jobs=-1)

#%%

# fiting model and finding best parameters 
gs_model = gs.fit(cgMLST_train, labels_train)

# mean performance results for the different parameters
performance_results_Trainingdata = pd.DataFrame(gs_model.cv_results_)
performance_results_Trainingdata = performance_results_Trainingdata[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
#performance_results_Trainingdata.to_csv("performanceTrainingdata_no_svm.csv", index=False)

# best model
print(gs_model.best_params_)
print(gs_model.best_score_)
clf = gs_model.best_estimator_

#%% 

# predicting 
labelno_predict = clf.predict(cgMLST_test)
source_predict=[label_dict[x] for x in labelno_predict]

#%% 

# performance metrics test
performance_report_test = classification_report(
            labels_test,
            labelno_predict,
            target_names=label_dict.values())

print(performance_report_test)

conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labels_test,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical')
conf_matrix.ax_.set_title("Conf. matrix")

#%%

# dataframe for the probabilityes predicted
source_true=[label_dict[x] for x in labels_test]
true_labels = [list(source_true)]
predictions = [list(source_predict)]
df_input = true_labels + predictions  
column_headers = ["true source","prediction"]
column_headers += ["probability_{}".format(label_dict[x])for x in range(len(label_dict.keys()))]

probability_df = pd.DataFrame(dict(zip(column_headers, df_input))).round(decimals=3)

# saving performance result test data
probability_df.to_csv("probability_test_no_svm.csv", index=False)
