# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
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
# feature selection based on mutual information
# percentile best features
percentile_threshold = 10
pBest= SelectPercentile(score_func=partial(mutual_info_classif, discrete_features=True, random_state=3), percentile=percentile_threshold)

# reducing train to p-best features
cgMLST_train_pBestReduced = pBest.fit_transform(cgMLST_train, labels_train)

#%% 

# setup for random forest model
RF_model = RandomForestClassifier(random_state=2)

# parameters
param_grid_RF = [{'n_estimators': [300, 400, 500, 600, 700, 800], 'criterion': ['gini']}]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

# gridsearch for best parameters 5-fold cross validation
gs_RF = GridSearchCV(estimator=RF_model, 
                  param_grid=param_grid_RF, 
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}), 
                  cv=cv,
                  refit='weighted_f1',
                  return_train_score=True,
                  n_jobs=-1)

#%%

# fiting model and finding best parameters 
gs_model_RF = gs_RF.fit(cgMLST_train_pBestReduced, labels_train)

# mean performance results for the different parameters
performanceResults_trainingdata = pd.DataFrame(gs_model_RF.cv_results_)
performanceResults_trainingdata = performanceResults_trainingdata[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
# performanceResults_trainingdata.to_csv("performanceTrainingdata_no_RF.csv", index=False)

# best model
print(gs_model_RF.best_params_)
print(gs_model_RF.best_score_)
clf_RF = gs_model_RF.best_estimator_

#%% 

# feature reduction test set
cgMLST_test_pBestReduced = pBest.transform(cgMLST_test)

# predicting 
proba_predict = clf_RF.predict_proba(cgMLST_test_pBestReduced)
labelno_predict = list(np.argmax(proba_predict, axis = 1))
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
proba_predict = list(proba_predict.T)
predictions += [list(x) for x in proba_predict]
df_input = labels_true + predictions  
column_headers = ["true source","prediction"]
column_headers += ["probability_{}".format(label_dict[x])for x in range(len(label_dict.keys()))]

probability_df = pd.DataFrame(dict(zip(column_headers, df_input))).round(decimals=3)

# saving performance result test data
# probability_df.to_csv("probability_test_no_RF.csv", index=False)
