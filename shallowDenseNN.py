# -*- coding: utf-8 -*-


#imports
import numpy as np
import pandas as pd
from functools import partial
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

#%%

# importing cleaned data
cleaned_data = pd.read_csv("cleaned_data_forML.csv")

#%%

# spliting source labels, cgmlst-data and SRA id-number
# (assuming SRA_no and Source is first and last column)
cgMLST_data = cleaned_data.iloc[:, 1:-1]
labels_txt = cleaned_data.Source
sample_id = cleaned_data.SRA_no

# encode lables as integers first and then as one-hot encoded dummies
encoder = LabelEncoder()
labels_int = encoder.fit_transform(labels_txt)
labels = OneHotEncoder(sparse_output=False).fit_transform(labels_int.reshape(-1, 1))

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

# setup for shallow dense neural network
ShallowDense_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, input_dim=1734),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ]
)

ShallowDense_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy", tf.keras.metrics.F1Score(average='weighted')])
ShallowDense_model.summary()

#%%

# feature selection based on mutual information
# percentile best features (10, 20, 30, 40, 50)
percentile_threshold = 10  
pBest= SelectPercentile(score_func=partial(mutual_info_classif, discrete_features=True, random_state=3), percentile=percentile_threshold)

# reducing train to p-best features
cgMLST_train_pBestReduced = pBest.fit_transform(cgMLST_train, labels_train)

#%%

# splitting trainigdata to get validation data for model training (20% validation data)
x_train, x_val, y_train, y_val = train_test_split(
        cgMLST_train,
        labels_train,
        test_size=0.20,
        stratify=labels_train,
        random_state=3)

#%%

ShallowDense_model_history = ShallowDense_model.fit(x_train, y_train, batch_size=541, epochs=200, validation_data=(x_val, y_val),
                                    callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50))

#%%

# feature reduction test set
cgMLST_test_pBestReduced = pBest.transform(cgMLST_test)

# get accuracy and f1-weighted scores
test_results = ShallowDense_model.evaluate(cgMLST_test_pBestReduced, labels_test, return_dict=True)
print(test_results)

# predicting test set
test_predict = ShallowDense_model.predict(cgMLST_test)
# np.savetxt(X= test_predict, fname="shallowDense_testPredict.csv", delimiter=",")

#%%

# probabilities test into source names 
proba_predict =  np.loadtxt("shallowDense_testPredict.csv", delimiter=",")
labelno_predict = list(np.argmax(proba_predict, axis = 1))
source_predict=[label_dict[x] for x in labelno_predict]

# true sources in test into source names
labelno_true = list(np.argmax(labels_test, axis = 1))
source_true=[label_dict[x] for x in labelno_true]

#%% 

# performance metrics for test prediction
performanceReport_testdata = classification_report(
            labelno_true,
            labelno_predict,
            target_names=label_dict.values())

print(performanceReport_testdata)

# confusionmatrix
conf_matrix = ConfusionMatrixDisplay.from_predictions(
            labelno_true,
            labelno_predict,
            display_labels=label_dict.values(),
            xticks_rotation= 'vertical')
conf_matrix.ax_.set_title("Conf. matrix SDnn-p")

#%%

# dataframe for the probabilityes predicted
labels_true = [list(source_true)]
predictions = [list(source_predict)]
proba_predictlst = list(proba_predict.T)
predictions += [list(x) for x in proba_predictlst]
df_input = labels_true + predictions  
column_headers = ["true source","prediction"]
column_headers += ["probability_{}".format(label_dict[x])for x in range(len(label_dict.keys()))]

probability_df = pd.DataFrame(dict(zip(column_headers, df_input))).round(decimals=3)

# saving performance result test data
#probability_df.to_csv("probability_test.csv", index=False)
