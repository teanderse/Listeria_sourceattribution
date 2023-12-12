# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

# computing mutual information for columns in train with classes in labels
mutual_info = mutual_info_classif(cgMLST_train, labels_train, random_state=3)
mutual_info = pd.Series(mutual_info)
mutual_info.index = cgMLST_train.columns
mutual_info.sort_values(ascending=False).head()

# saving mutuak information calculation for features in train
# mutual_info.to_csv("mutualInfo_trainingdata.csv", index=True)

#%%
np.random.seed(3)
# feature selection based on mutual information
# percentile best features
percentile_threshold = 50
pBest= SelectPercentile(mutual_info_classif, percentile=percentile_threshold)

# reducing train to p-best features
cgMLST_train_pBestReduced = pBest.fit_transform(cgMLST_train, labels_train)

#%%
# setup for support vector clasifier 
SVM_model = SVC(random_state=2)
# SVC with pipeline fro scaling
SVM_pipe = make_pipeline(StandardScaler(),
                         SVC(random_state=2))

# parameter range for C
param_rangeC  = [0.01, 0.03, 0.05, 0.07,  0.1, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
# parameter range for gamma for scaling of the rbf-kernel
param_rangeG = [0.0001, 0.001,0.005, 0.01, 0.015, 0.1, 1.0, 5.0]   

# parameters
# add svc__ for SVM_pipe   
param_grid_SVM = [{'svc__C': param_rangeC, 'svc__gamma': param_rangeG, 'svc__kernel': ['rbf']}]

# Estimator: SVM_pipe = scaling, SVM_model = no scaling
gs_SVM = GridSearchCV(estimator=SVM_pipe, 
                  param_grid=param_grid_SVM, 
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}), 
                  cv=5,
                  refit='weighted_f1',
                  return_train_score=True,
                  n_jobs=-1)

#%%

# fiting model and finding best parameters 
gs_model_SVM = gs_SVM.fit(cgMLST_train_pBestReduced, labels_train)

# mean performance results for the different parameters
performanceResults_trainingdata = pd.DataFrame(gs_model_SVM.cv_results_)
performanceResults_trainingdata = performanceResults_trainingdata[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
#performanceResults_trainingdata.to_csv("performanceTrainingdata_no_svm.csv", index=False)

# best model
print(gs_model_SVM.best_params_)
print(gs_model_SVM.best_score_)
clf_SVM = gs_model_SVM.best_estimator_

#%% 

# predicting 
labelno_predict = clf_SVM.predict(cgMLST_test)
source_predict=[label_dict[x] for x in labelno_predict]

#%% 

# performance metrics test
performanceReport_testdata = classification_report(
            labels_test,
            labelno_predict,
            target_names=label_dict.values())

print(performanceReport_testdata)

conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labels_test,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical')
conf_matrix.ax_.set_title("Conf. matrix")

#%%

# dataframe for the probabilityes predicted
source_true=[label_dict[x] for x in labels_test]
labels_true = [list(source_true)]
predictions = [list(source_predict)]
df_input = labels_true + predictions  
column_headers = ["true source","prediction"]
column_headers += ["probability_{}".format(label_dict[x])for x in range(len(label_dict.keys()))]

probability_df = pd.DataFrame(dict(zip(column_headers, df_input))).round(decimals=3)

# saving performance result test data
probability_df.to_csv("probability_test_no_svm.csv", index=False)
