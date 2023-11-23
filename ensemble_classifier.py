# -*- coding: utf-8 -*-


# imports
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel

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
# under construction
#-------------------------------------------------------------------------

# feature reduction

# removing no variance features
unique_train = cgMLST_train.nunique()
no_var = [i for i, v in enumerate(unique_train) if v == 1]
removed_features_noVar = unique_train.index[no_var].tolist()
cgMLST_train_reduced = cgMLST_train.drop(cgMLST_train.columns[no_var], axis=1)

# finding low variance features by threshold for percent unique values
percentVar_threshold = 2
low_varPercent = [i for i,v in enumerate(unique_train) if (float(v)/cgMLST_train.shape[0]*100) < percentVar_threshold]
removed_features_percent = unique_train.index[low_varPercent].tolist()

# finding low variance features by ratio between the frequency of the two most common values
cgMLST_train_reduced_dummy = cgMLST_train_reduced.apply(lambda x: x.value_counts().values[0:2]).T
cgMLST_train_reduced_dummy[2] = cgMLST_train_reduced_dummy.apply(lambda x: x[0]/x[1], axis=1)
percentRatio_threshold = 95/5
low_varRatio = [i for i,v in enumerate(cgMLST_train_reduced_dummy[2]) if (round(v, 2)) > percentRatio_threshold]
removed_features_ratio = cgMLST_train_reduced_dummy.index[low_varRatio].tolist()

# removing features with both low percente unique values and ratio between value frquencies
removed_features = list(set(removed_features_percent) & set(removed_features_ratio))
cgMLST_train_reduced = cgMLST_train_reduced.drop(columns=removed_features)

# Removing features fron test set 

#-------------------------------------------------------------------------
#%%

# setup for random forest model
model = RandomForestClassifier(random_state=2)

# parameters
param_grid   = [{'n_estimators': [300, 400, 500], 'class_weight':['balanced', None], 'criterion': ['gini']}]

# gridsearch for best parameters 5-fold cross validation
gs = GridSearchCV(estimator=model, 
                  param_grid=param_grid, 
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}), 
                  cv=5,
                  refit='weighted_f1',
                  return_train_score=True,
                  n_jobs=-1)

#%%

# fiting model and finding best parameters 
gs_model1 = gs.fit(cgMLST_train_reduced, labels_train)

# mean performance results for the different parameters
performance_results1 = pd.DataFrame(gs_model1.cv_results_)
performance_results1 = performance_results1[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
# performance_results1.to_csv("performanceTrainingdata_2_95_5.csv", index=False)

# best model
print(gs_model1.best_params_)
print(gs_model1.best_score_)
clf1 = gs_model1.best_estimator_

#%%

# under construction
#-------------------------------------------------------------------------

# feature reduction based on feature importanse RF
feature_reductionRF = SelectFromModel(clf1)
feature_reductionRF.fit(cgMLST_train_reduced, labels_train)

selected_features = cgMLST_train_reduced.columns[(feature_reductionRF.get_support())]
len(selected_features)

cgMLST_train_reducedRF = feature_reductionRF.transform(cgMLST_train_reduced) 

# Removing features fron test set

#--------------------------------------------------------------------------

#%%

# refiting model after feature reduction and finding best parameters 
gs_model2 = gs.fit(cgMLST_train_reducedRF, labels_train)

# mean performance results for the different parameters
performance_results2 = pd.DataFrame(gs_model2.cv_results_)
performance_results2 = performance_results2[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
# performance_results2.to_csv("performanceTrainingdata_RF498.csv", index=False)

# best model
print(gs_model2.best_params_)
print(gs_model2.best_score_)
clf2 = gs_model2.best_estimator_

#%% 

# predicting 
proba_predict = clf2.predict_proba(cgMLST_test)
labelno_predict = list(np.argmax(proba_predict, axis = 1))
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
proba_predict = list(proba_predict.T)
predictions += [list(x) for x in proba_predict]
df_input = true_labels + predictions  
column_headers = ["true source","prediction"]
column_headers += ["probability_{}".format(label_dict[x])for x in range(len(label_dict.keys()))]

probability_df = pd.DataFrame(dict(zip(column_headers, df_input))).round(decimals=3)

# saving performance result test data
# probability_df.to_csv("probability_test_no.csv", index=False)
