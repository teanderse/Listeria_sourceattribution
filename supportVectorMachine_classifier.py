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

# importing cleaned data for cgMLST or wgMLST
MLST_type = "cg" # cg or wg
cleaned_data = pd.read_csv(f"cleaned_data_forML/{MLST_type}MLSTcleaned_data_forML.csv")

#%%

# spliting source labels, MLST-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
MLST_data = cleaned_data.iloc[:, 1:-1]
labels = cleaned_data.Source
sample_id = cleaned_data.SRA_no

# encode source lables as integers
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# saving source label integer and source name in dictionary
label_dict = dict(zip((encoder.transform(encoder.classes_)), encoder.classes_ ))

#%%

# split randomly into training(70%) and testing(30%), stratified by source label, with seed for reproducibility
MLST_train, MLST_test, labels_train, labels_test = train_test_split(
        MLST_data,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=3)

#%%
# setup for support vector clasifier 
SVMno_model = SVC()
# setup for support vector pipeline with scaling
SVM_pipe = make_pipeline(StandardScaler(),
                         SVC())

# hyperparameter range for C (cost of missclassification)
param_rangeC  = [1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0]
# hyperparameter range for gamma for scaling of the rbf-kernel
param_rangeG = [0.0005, 0.001, 0.002, 0.003, 0.005]   

# hyperparameter grid
# for SVM_model with no scaling
param_grid_SVMno = [{'C': param_rangeC, 'gamma': param_rangeG, 'kernel': ['rbf']}]
# for SVM_pipe with scaling 
param_grid_SVM = [{'svc__C': param_rangeC, 'svc__gamma': param_rangeG, 'svc__kernel': ['rbf']}]

# 5-fold cross validation with 10 repeats
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=3)

# chosing model with/without scaling and setup for hyperparameter grid 
model = SVM_pipe      # SVMno_model for no scaling or SVM_pipe for scaling
grid = param_grid_SVM # param_grid_SVMno for SVMno_model or param_grid_SVM for SVM_pipe

# grid search for search of optimal hyperparameters
gs_SVM = GridSearchCV(estimator=model, 
                  param_grid=grid, 
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}), 
                  cv=cv,
                  refit='weighted_f1',
                  return_train_score=True,
                  n_jobs=-1)

#%%

# feature selection only done for cgMLST data

# feature selection based on mutual information
# percentile best features 
percentile_threshold = 50 #(10, 20, 30, 40 or 50)
pBest= SelectPercentile(score_func=partial(mutual_info_classif, discrete_features=True), 
                        percentile=percentile_threshold)

# finding and reducing training set to p-best features
cgMLST_train_pBestReduced = pBest.fit_transform(MLST_train, labels_train)
# reducing test set based on calculation for best features done on training set
cgMLST_test_pBestReduced = pBest.transform(MLST_test)

#%%

# fitting model to MLST_train for all features or cgMLST_train_pBestReduced for selected features in cgMLST data
# finding optimal hyperparameters with grid search  
gs_model_SVM = gs_SVM.fit(MLST_train, labels_train)

# mean performance results for the different hyperparameters tested in grid search
performanceResults_trainingdata = pd.DataFrame(gs_model_SVM.cv_results_)
performanceResults_trainingdata = performanceResults_trainingdata[['params','mean_test_weighted_f1', 'rank_test_weighted_f1', 
                   'mean_test_macro_f1', 'rank_test_macro_f1',
                   'mean_test_accurcacy', 'rank_test_accurcacy']]

# saving performance result training data
performanceResults_trainingdata.to_csv(f"performanceTrainingdata_SVM_{MLST_type}MLST.csv", index=False)

# defining optimal hyperparameter and score for best model 
clf_SVM = gs_model_SVM.best_estimator_
print(gs_model_SVM.best_params_)
print(gs_model_SVM.best_score_)

#%% 

# predicting source for test data using best model on MLST_test for all features or cgMLST_test_pBestReduced for selected features in cgMLST data
labelno_predict = clf_SVM.predict(MLST_test)
source_predict=[label_dict[x] for x in labelno_predict]

#%% 

# defining settings for features to save test performance data
# percentile best features 10%, 20%, 30%, 40%, 50% or all 
feature = "all"
percent = f"{feature} features"

# performance metrics for test prediction
performanceReport_testdata = classification_report(
            labels_test,
            labelno_predict,
            target_names=label_dict.values(),
            output_dict = True)

performanceReport_testdata_df = pd.DataFrame.from_dict(performanceReport_testdata)
# saving performance metrics for test predictions
performanceReport_testdata_df.to_csv(f"{feature}_SVM_performanceReport_testdata__{MLST_type}MLSTdf.csv")

# confusionmatrix for predictions
conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labels_test,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical',
            cmap='Greens')
conf_matrix.ax_.set_title(f"Conf. matrix SVM/scale {percent} {MLST_type}MLST")
# saving consution matrix
conf_matrix.figure_.savefig(f'{feature}_confmatSVM_{MLST_type}MLST.png')

#%% 

# using model to predicting sources for clinical isolates
# reading in the data for the clinical isolates
clinical_isolates = pd.read_csv(f"cleaned_clinical/{MLST_type}MLST_clinical_samples.csv")
clinical_data = clinical_isolates.drop(['SRA_no', 'Source'], axis=1)
clinical_id = clinical_isolates.SRA_no

# predicting source for clinical isolates
labelno_predict_clinical = clf_SVM.predict(clinical_data)
source_predict_clinical=[label_dict[x] for x in labelno_predict_clinical]

# dataframe for the predictions
probability_clinical_df = pd.DataFrame(source_predict_clinical, columns=["prediction"])
probability_clinical_df.insert(0, "SRA_no", clinical_id)

# saving source predicted for clinical isolates
probability_clinical_df.to_csv(f"probability_clinical_SVM_{MLST_type}MLST.csv", index=False)
