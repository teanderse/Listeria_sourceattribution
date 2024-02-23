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
labels_encoded = encoder.fit_transform(labels)

# saving label integer and source name in dictionary
label_dict = dict(zip((encoder.transform(encoder.classes_)), encoder.classes_ ))

#%%

# split randomly into training(70%) and testing(30%)
cgMLST_train, cgMLST_test, labels_train, labels_test = train_test_split(
        cgMLST_data,
        labels_encoded,
        test_size=0.30,
        stratify=labels_encoded,
        random_state=3)

#%% 

# feature selection based on mutual information
# percentile best features (10, 20, 30, 40, 50)
percentile_threshold = 10  
pBest= SelectPercentile(score_func=partial(mutual_info_classif, discrete_features=True, random_state=3), percentile=percentile_threshold)

# reducing train to p-best features
cgMLST_train_pBestReduced = pBest.fit_transform(cgMLST_train, labels_train)
# feature reduction test set
cgMLST_test_pBestReduced = pBest.transform(cgMLST_test)

#%%

# one hot encoding the labels
labels_test = OneHotEncoder(sparse_output=False).fit_transform(labels_test.reshape(-1, 1))
labels_train = OneHotEncoder(sparse_output=False).fit_transform(labels_train.reshape(-1, 1))

#%%

# adding metrics for performance measure
f1_macro = tf.keras.metrics.F1Score(average='macro', name='f1_macro')
f1_weighted =tf.keras.metrics.F1Score(average='weighted', name='f1_weighted')

#%%

# function for making neural network models and finding best hyperparameters with gridsearch
def create_shallowDenseNN(input_dim, neurons, dropout_rate ):
  ShallowDense_model = tf.keras.Sequential()
  ShallowDense_model.add(tf.keras.layers.Dense(neurons, input_dim=input_dim, activation="relu"))
  ShallowDense_model.add(tf.keras.layers.Dropout(dropout_rate))
  ShallowDense_model.add(tf.keras.layers.Dense(5, activation="softmax"))

  return ShallowDense_model

# set up for shallow dense neural network
# input dimentions for different feature selected data sets
all_ = 1734
# p10 = 174
# p20 = 347
# p30 = 520
# p40 = 694
# p50 = 867
ShallowDense_model = KerasClassifier(model=create_shallowDenseNN, input_dim=all_, loss="categorical_crossentropy",
                                     optimizer=tf.keras.optimizers.Adam,
                                     metrics=["accuracy", f1_weighted, f1_macro], epochs=100, batch_size=676,
                                     callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50))

# tune number of nodes, dropout rate and learning rate
# hyperparameter for SDNN_model
learning_rate = [0.0005, 0.0001]
dropout_rate = [0.3, 0.4]
nodes = [75, 78, 80]
param_grid_SDNN = [{'model__neurons':nodes, 'optimizer__learning_rate': learning_rate, 'model__dropout_rate':dropout_rate}]

# gridsearch for best parameter search run 5 times for
gs_SDNN = GridSearchCV(estimator=ShallowDense_model,
                  param_grid=param_grid_SDNN,
                  scoring=({'weighted_f1':'f1_weighted', 'macro_f1':'f1_macro', 'accurcacy':'accuracy'}),
                  cv=5,
                  refit='weighted_f1',
                  return_train_score=True)

# fiting model to cgMLST_train for all features and cgMLST_train_pBestReduced for selected features
gs_model_SDNN = gs_SDNN.fit(cgMLST_train, labels_train)

print("Best fi weighted: %f using %s" % (gs_model_SDNN.best_score_, gs_model_SDNN.best_params_))

#%%

# best hyperparmeters set as found in grid search
# all features
neuron_no = 78
dropout_r = 0.2
learning_r = 0.0001 # all features
# learning_r = 0.001 # selected features
# input dimentions for different feature selected data sets
all_ = 1734
# p10 = 174
# p20 = 347
# p30 = 520
# p40 = 694
# p50 = 867


# empty lists to append performance metrics
accu_list_all = list()
f1W_list_all = list()
f1M_list_all = list()
test_results= []

# looping over trainig and testing 30 times
for i in range(1,31):
  ShallowDense_model_optimized = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(neuron_no, input_dim=all_, activation="relu"),
        tf.keras.layers.Dropout(dropout_r),
        tf.keras.layers.Dense(5, activation="softmax")
    ]
  )

  ShallowDense_model_optimized.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_r), metrics=["accuracy", f1_weighted, f1_macro])
  ShallowDense_model_optimized_history = ShallowDense_model_optimized.fit(cgMLST_train, labels_train, batch_size=676, epochs=100, callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8))
  # appending training performanse metrics
  accu = list(ShallowDense_model_optimized_history.history['accuracy'])[-1]
  accu_list_all.append(accu)
  f1W = list(ShallowDense_model_optimized_history.history['f1_weighted'])[-1]
  f1W_list_all.append(f1W)
  f1M = list(ShallowDense_model_optimized_history.history['f1_macro'])[-1]
  f1M_list_all.append(f1M)
    
  # appending testing performance metrics and saving predictions
  test_res = ShallowDense_model_optimized.evaluate(cgMLST_test, labels_test, return_dict=True)
  test_results.append(test_res)
  test_pred = ShallowDense_model_optimized.predict(cgMLST_test)
  # np.savetxt(X= test_pred, fname=f"/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/all_shallowDense_testPredict{i}.csv", delimiter=",")

#%%

# Saving performance metrics for 30 repeats

# np.savetxt(X= f1W_list_all, fname=f"/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/f1W_train_all.csv", delimiter=",")
# np.savetxt(X= f1M_list_all, fname=f"/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/f1M_train_all.csv", delimiter=",")
# np.savetxt(X= accu_list_all, fname=f"/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/accu_train_all.csv", delimiter=",")
# np.savetxt(X= test_results, fname=f"/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/testResults_all.csv", delimiter=",")

# test_result_df = pd.DataFrame.from_dict(test_results)
# test_result_df.to_csv("/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/testResults_all.csv")


#%%

# saving performance report for 30 repeats
for i in range(1,31):
  # probabilities test into source names
  proba_predict = np.loadtxt(f'/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/all_shallowDense_testPredict{i}.csv', delimiter=',')
  labelno_predict = list(np.argmax(proba_predict, axis = 1))
  source_predict=[label_dict[x] for x in labelno_predict]

  # true sources in test into source names
  labelno_true = list(np.argmax(labels_test, axis = 1))
  source_true=[label_dict[x] for x in labelno_true]

  performanceReport_testdata = classification_report(
              labelno_true,
              labelno_predict,
              target_names=label_dict.values(),
              output_dict = True)

  performanceReport_testdata_df = pd.DataFrame.from_dict(performanceReport_testdata)
  #performanceReport_testdata_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/cgMLST_data/all_shallowDense_testPred/all_performanceReport_testdata_df_{i}.csv")

#%% 

# confusionmatrix
conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labelno_true,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical',
            cmap='Greens')
conf_matrix.ax_.set_title("Conf. matrix SDnn -%")

