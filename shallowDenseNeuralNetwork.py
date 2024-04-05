# -*- coding: utf-8 -*-


#imports
import numpy as np
import pandas as pd
from functools import partial
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from scikeras.wrappers import KerasClassifier

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

# encode lables as integers
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# saving source label integer and source name in dictionary
label_dict = dict(zip((encoder.transform(encoder.classes_)), encoder.classes_ ))

#%%

# split randomly into training(70%) and testing(30%), stratified by class label, with seed for reproducibility
MLST_train, MLST_test, labels_train, labels_test = train_test_split(
        MLST_data,
        labels_encoded,
        test_size=0.30,
        stratify=labels_encoded,
        random_state=3)

#%% 

# feature selection only done for cgMLST data

# feature selection based on mutual information, with seed for reproducibility
# percentile best features
percentile_threshold = 50  #(10, 20, 30, 40 or 50)
pBest= SelectPercentile(score_func=partial(mutual_info_classif, discrete_features=True),
                        percentile=percentile_threshold)

# finding and reducing training set to p-best features
cgMLST_train_pBestReduced = pBest.fit_transform(MLST_train, labels_train)
# reducing test set based on calculation for best features done on training set
cgMLST_test_pBestReduced = pBest.transform(MLST_test)

#%%

# one-hot encoding the source labels in both training and test data sets
oh_encoder = OneHotEncoder(sparse_output=False)
labels_train = oh_encoder.fit_transform(labels_train.reshape(-1, 1))
labels_test = oh_encoder.transform(labels_test.reshape(-1, 1))

#%%

# adding metrics for performance measure
f1_macro = tf.keras.metrics.F1Score(average='macro', name='f1_macro')
f1_weighted =tf.keras.metrics.F1Score(average='weighted', name='f1_weighted')

#%%

# function for making shallow dense neural network models and finding optimal hyperparameters with gridsearch
def create_shallowDenseNN(input_dim, neurons, dropout_rate ):
  ShallowDense_model = tf.keras.Sequential()
  ShallowDense_model.add(tf.keras.layers.Dense(neurons, input_dim=input_dim, activation="relu"))
  ShallowDense_model.add(tf.keras.layers.Dropout(dropout_rate))
  ShallowDense_model.add(tf.keras.layers.Dense(5, activation="softmax"))

  return ShallowDense_model

# set up for shallow dense neural network model
# input dimentions for wgMLST data or different feature selected data sets of cgMLST data
wgMLST = 2496
all_ = 1734
p10 = 174
p20 = 347
p30 = 520
p40 = 694
p50 = 867

ShallowDense_model = KerasClassifier(model=create_shallowDenseNN, input_dim=all_, loss="categorical_crossentropy",
                                     optimizer=tf.keras.optimizers.Adam,
                                     metrics=["accuracy", f1_weighted, f1_macro], epochs=200, batch_size=676,
                                     callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10))

# hyperparameter for ShallowDense_model
# tune number of nodes in hidden layer, dropout rate and learning rate
learning_rate = [0.001, 0.0001]
dropout_rate = [0.2, 0.3]
nodes = [70, 75, 80]

# hyperparameter grid
param_grid_SDNN = [{'model__neurons':nodes, 'optimizer__learning_rate': learning_rate, 'model__dropout_rate':dropout_rate}]

# grid search for search of optimal hyperparameters, with 5-fold cross validation
gs_SDNN = GridSearchCV(estimator=ShallowDense_model,
                  param_grid=param_grid_SDNN,
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}),
                  cv=5,
                  refit='weighted_f1',
                  return_train_score=True)

# fitting model to MLST_train for all features or cgMLST_train_pBestReduced for selected features in cgMLST data
# finding opyimal hyperparameters with grid search 
gs_model_SDNN = gs_SDNN.fit(MLST_train, labels_train)

# defining optimal hyperparameter and score for best model 
print(gs_model_SDNN.best_params_)
print(gs_model_SDNN.best_score_) 

#%%

# mean performance results for the different hyperparameters tested in grid searc
performanceResults_trainingdata = pd.DataFrame(gs_model_SDNN.cv_results_)
performanceResults_trainingdata = performanceResults_trainingdata[['params','mean_test_weighted_f1']]
# saving performance result training data 
performanceResults_trainingdata.to_csv(f"performanceTrainingdata_SDNN_{MLST_type}MLST.csv.csv")

#%%

# optimal hyperparmeters set as they were defind by the grid search
# (here shown with optimal hyperparameters for the cgMLST training data set with all features)
neuron_no = 80 
dropout_r = 0.2
learning_r = 0.0001 

# input dimentions for wgMLST data or different feature selected data sets of cgMLST data
wgMLST = 2496
all_ = 1734
p10 = 174
p20 = 347
p30 = 520
p40 = 694
p50 = 867

# looping over trainig and testing of model 30 times
for i in range(1,31):
  ShallowDense_model_optimized = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(neuron_no, input_dim=all_, activation="relu"),
        tf.keras.layers.Dropout(dropout_r),
        tf.keras.layers.Dense(5, activation="softmax")
    ]
  )

  ShallowDense_model_optimized.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_r), metrics=["accuracy", f1_weighted, f1_macro])
  ShallowDense_model_optimized_history = ShallowDense_model_optimized.fit(MLST_train, labels_train, batch_size=676, epochs=200, callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10))
  
  # predicting source for test data using best model on MLST_test for all features or cgMLST_test_pBestReduced for selected features in cgMLST data
  # saving predictions for each iteration of the loop
  test_pred = ShallowDense_model_optimized.predict(MLST_test)
  np.savetxt(X= test_pred, fname=f"shallowDense_testPredict{i}.csv", delimiter=",")

#%%

# making and saving performance report with metrics for the 30 predictions made by the best model
for i in range(1,31):
  # transforming the predictions into integers and source names
  proba_predict = np.loadtxt(f'shallowDense_testPredict{i}.csv', delimiter=',')
  labelno_predict = list(np.argmax(proba_predict, axis = 1))
  source_predict=[label_dict[x] for x in labelno_predict]

  # transforming the true, one-hot encoded test labels for source back to integers
  labelno_true = list(np.argmax(labels_test, axis = 1))
  
  # performance metrics for test prediction 
  performanceReport_testdata = classification_report(
              labelno_true,
              labelno_predict,
              target_names=label_dict.values(),
              output_dict = True)

  performanceReport_testdata_df = pd.DataFrame.from_dict(performanceReport_testdata)
  # saving performance metrics for test predictions
  performanceReport_testdata_df.to_csv(f"performanceReport_testdata_df_{i}.csv")

#%% 

# defining settings for features to save test performance as confusion matix
# percentile best features 10%, 20%, 30%, 40%, 50% or all 
feature = "all"
percent = f"{feature} features" 

# confusionmatrix for predictions
conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labelno_true,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical',
            cmap='Greens')
conf_matrix.ax_.set_title("Conf. matrix SDNN {percent} {MLST_type}MLST")
# saving confusion matrix
conf_matrix.figure_.savefig(f'{feature}_confmatSDNN_{MLST_type}MLST.png')

#%%

# using model to predicting sources for clinical isolates
# reading in the data for the clinical isolates
clinical_isolates = pd.read_csv(f"cleaned_clinical/{MLST_type}MLST_clinical_samples.csv")
clinical_data = clinical_isolates.drop(['SRA_no', 'Source'], axis=1)
clinical_id = clinical_isolates.SRA_no

# predicting source for clinical isolates
proba_predict_clinical = test_pred = ShallowDense_model_optimized.predict(clinical_data)
labelno_predict_clinical = list(np.argmax(proba_predict_clinical, axis = 1))
source_predict_clinical=[label_dict[x] for x in labelno_predict_clinical]

# dataframe for the predictions
probability_clinical_df = pd.DataFrame(source_predict_clinical, columns=["prediction"])
probability_clinical_df.insert(0, "SRA_no", clinical_id)

# saving source predicted for clinical isolates
probability_clinical_df.to_csv(f"probability_clinical_SDNN_{MLST_type}MLST.csv.csv", index=False)
