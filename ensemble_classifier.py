# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# from sklearn.feature_selection import SelectFromModel

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

# computing shannon entropy for columns in train
cgMLST_train_count = cgMLST_train.apply(lambda x: x.value_counts())
cgMLST_train_count.fillna(0, inplace=True)
cgMLST_train_abundance = cgMLST_train_count.apply(lambda x: x/(cgMLST_train_count.shape[0]))
s_entropy = cgMLST_train_abundance.apply(lambda x: entropy(x))

# defining features with low entropy
entropy_threshold = 0
low_entropy = [i for i,v in enumerate(s_entropy) if v == entropy_threshold]
lowEntropy_features = s_entropy.index[low_entropy].tolist()

#%%

# Removing features with low diversity from train and test set
cgMLST_train_divReduced = cgMLST_train.drop(columns=lowEntropy_features)
cgMLST_test_divReduced = cgMLST_test.drop(columns=lowEntropy_features)

#%%

# setup for random forest model
RF_model = RandomForestClassifier(random_state=2)

# parameters
param_grid_RF = [{'n_estimators': [300, 400, 500, 600, 700, 800], 'class_weight':['balanced', None], 'criterion': ['gini']}]

# gridsearch for best parameters 5-fold cross validation
gs_RF = GridSearchCV(estimator=RF_model, 
                  param_grid=param_grid_RF, 
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}), 
                  cv=5,
                  refit='weighted_f1',
                  return_train_score=True,
                  n_jobs=-1)

#%%

# fiting model and finding best parameters 
gs_model_RF = gs_RF.fit(cgMLST_train, labels_train)

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

# under construction
#-------------------------------------------------------------------------

# # feature selection based on feature importanse RF
# feature_importance = gs_model_RF.best_estimator_.feature_importances_ 
# # selecting features based on importance
# important_features

# # Removing features from test train set
# cgMLST_train_reduced = cgMLST_train_divReduced.columns[important_features]
# cgMLST_test_reduced = cgMLST_test_divReduced.columns[important_features]

# # refiting model after feature selection and finding best parameters 
# gs_model_RF2 = gs_RF.fit(cgMLST_train_reduced, labels_train)

# # mean performance results for the different parameters
# performanceResults_trainingdata2 = pd.DataFrame(gs_model_RF2.cv_results_)
# performanceResults_trainingdata2 = performanceResults_trainingdata2[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
#                    'mean_test_macro_f1', 'rank_test_macro_f1',
#                    'mean_test_accurcacy', 'rank_test_accurcacy']]

# # saving performance result training data
# # performanceResults_trainingdata2.to_csv("performanceTrainingdata_RF498.csv", index=False)

# # best model
# print(gs_model_RF2.best_params_)
# print(gs_model_RF2.best_score_)
# clf_RF2 = gs_model_RF2.best_estimator_

#--------------------------------------------------------------------------

#%% 

# predicting 
proba_predict = clf_RF.predict_proba(cgMLST_test)
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
probability_df.to_csv("probability_test_no_RF.csv", index=False)
