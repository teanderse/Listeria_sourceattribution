# -*- coding: utf-8 -*-


# imports
import pandas as pd
from functools import partial
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectPercentile 
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
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
# setup for support vector clasifier 
SVM_model = SVC()
# setup for support vector pipeline with scaling
SVM_pipe = make_pipeline(StandardScaler(),
                         SVC())

# parameter range for C (cost)
param_rangeC  = [1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0]
# parameter range for gamma for scaling of the rbf-kernel
param_rangeG = [0.0005, 0.001, 0.002, 0.003, 0.005]   

# parameters
# for SVM_model with no scaling
param_grid_SVM = [{'C': param_rangeC, 'gamma': param_rangeG, 'kernel': ['rbf']}]
# for SVM_pipe with scaling 
#param_grid_SVM = [{'svc__C': param_rangeC, 'svc__gamma': param_rangeG, 'svc__kernel': ['rbf']}]

# 5-fold cross validation with 10 repeats
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=3)

# gridsearch for best parameter search
# estimator: SVM_pipe = scaling, SVM_model = no scaling
gs_SVM = GridSearchCV(estimator=SVM_model, 
                  param_grid=param_grid_SVM, 
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
gs_model_SVM = gs_SVM.fit(cgMLST_train, labels_train)

# mean performance results for the different parameters
performanceResults_trainingdata = pd.DataFrame(gs_model_SVM.cv_results_)
performanceResults_trainingdata = performanceResults_trainingdata[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
# performanceResults_trainingdata.to_csv("performanceTrainingdata_SVM.csv", index=False)

# best model
clf_SVM = gs_model_SVM.best_estimator_
print(gs_model_SVM.best_params_)
print(gs_model_SVM.best_score_)

#%% 

# feature reduction test set
cgMLST_test_pBestReduced = pBest.transform(cgMLST_test)

# predicting test using best model on cgMLST_test for all features and cgMLST_test_pBestReduced for selected features
labelno_predict = clf_SVM.predict(cgMLST_test)
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
performanceReport_testdata_df.to_csv(f"{feature}_SVM_performanceReport_testdata_df.csv")

# confusionmatrix
conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labels_test,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical',
            cmap='Greens')
conf_matrix.ax_.set_title(f"Conf. matrix SVM noscale {percent}")
conf_matrix.figure_.savefig(f'{feature}_confmatSVM.png')

