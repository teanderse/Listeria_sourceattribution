# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:18:42 2024

@author: tessa
"""



# fitting model to training data set and finding best hyperparameters with grid search  
gs_model_RF = gs_RF.fit(MLST_train, labels_train)

# defining optimal model 
clf_RF = gs_model_RF.best_estimator_

# predicting sources in test data set with optimal model
proba_predict = clf_RF.predict_proba(MLST_test)



# fitting model to training data set and finding best hyperparameters with grid search  
gs_model_SVM = gs_SVM.fit(MLST_train, labels_train)

# defining optimal model 
clf_SVM = gs_model_SVM.best_estimator_

# predicting sources in test data set with optimal model
predict = clf_SVM.predict(MLST_test)


# fitting model to training data set and finding best hyperparameters with grid search 
gs_model_SDNN = gs_SDNN.fit(MLST_train, labels_train)


# fitting model to training data set with optimal hyperparameter values
ShallowDense_model_optimized.fit(MLST_train, labels_train, batch_size=676, epochs=200, callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10))

# predicting sources in test data set with optimal model
test_pred = ShallowDense_model_optimized.predict(MLST_test)
