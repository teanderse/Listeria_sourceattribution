# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from functools import partial
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

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

# setup for random forest classifier
RF_model = RandomForestClassifier(random_state=2)

# parameter for RF_model
param_grid_RF = [{'n_estimators': [300, 400, 500, 600, 700, 800], 'criterion': ['gini']}]

# 5-fold cross validation with 10 repeats
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=3)

# gridsearch for best parameter search
gs_RF = GridSearchCV(estimator=RF_model, 
                  param_grid=param_grid_RF, 
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}), 
                  cv=cv,
                  refit='weighted_f1',
                  return_train_score=True,
                  n_jobs=-1)

#%%

# feature selection based on mutual information
# percentile best features (10, 20, 30, 40, 50)
percentile_threshold = 50  
pBest= SelectPercentile(score_func=partial(mutual_info_classif, discrete_features=True, random_state=3), percentile=percentile_threshold)

# reducing train to p-best features
cgMLST_train_pBestReduced = pBest.fit_transform(cgMLST_train, labels_train)

#%%

# fiting model to cgMLST_train for all features and cgMLST_train_pBestReduced for selected features
# finding best hyperparameters  
gs_model_RF = gs_RF.fit(cgMLST_train, labels_train)

# mean performance results for the different parameters
performanceResults_trainingdata = pd.DataFrame(gs_model_RF.cv_results_)
performanceResults_trainingdata = performanceResults_trainingdata[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
#performanceResults_trainingdata.to_csv("performanceTrainingdata_RFmodel.csv", index=False)

# best model
clf_RF = gs_model_RF.best_estimator_
print(gs_model_RF.best_params_)
print(gs_model_RF.best_score_)

#%% 

# feature reduction test set
cgMLST_test_pBestReduced = pBest.transform(cgMLST_test)

# predicting test using best model on cgMLST_test for all features and cgMLST_test_pBestReduced for selected features
proba_predict = clf_RF.predict_proba(cgMLST_test)
labelno_predict = list(np.argmax(proba_predict, axis = 1))
source_predict=[label_dict[x] for x in labelno_predict]

#%% 

# percentile best features (10%, 20%, 30%, 40%, 50%), and all
feature = "all"
percent = "all features"

# performance metrics for test prediction
performanceReport_testdata = classification_report(
            labels_test,
            labelno_predict,
            target_names=label_dict.values(),
            output_dict = True)

performanceReport_testdata_df = pd.DataFrame.from_dict(performanceReport_testdata)
performanceReport_testdata_df.to_csv(f"{feature}_RF_performanceReport_testdata_df")

# confusionmatrix
conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labels_test,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical',
            cmap='Greens')
conf_matrix.ax_.set_title(f"Conf. matrix RF {percent}")
conf_matrix.figure_.savefig(f'{feature}_confmatRF.png')

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
# probability_df.to_csv(f"probability_test_RFmodel_{feature}.csv", index=False)
