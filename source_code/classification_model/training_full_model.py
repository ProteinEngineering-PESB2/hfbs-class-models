import pandas as pd
import sys

#for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

#for check overfitting
from sklearn.model_selection import cross_validate
import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

import json

#function to obtain metrics using the testing dataset
def get_performances(description, predict_label, real_label):
    accuracy_value = accuracy_score(real_label, predict_label)
    f1_score_value = f1_score(real_label, predict_label, average='weighted')
    precision_values = precision_score(real_label, predict_label, average='weighted')
    recall_values = recall_score(real_label, predict_label, average='weighted')

    row = [description, accuracy_value, f1_score_value, precision_values, recall_values]
    return row

#function to process average performance in cross val training process
def process_performance_cross_val(performances, keys):
    
    row_response = []
    for i in range(len(keys)):
        value = np.mean(performances[keys[i]])
        row_response.append(value)
    return row_response

#function to train a predictive model
def training_process(model, X_train, y_train, X_test, y_test, scores, cv_value, description, keys):
    print("Train model with cross validation")
    model.fit(X_train, y_train)
    response_cv = cross_validate(model, X_train, y_train, cv=cv_value, scoring=scores)
    performances_cv = process_performance_cross_val(response_cv, keys)

    print("Predict responses and make evaluation")
    responses_prediction = clf.predict(X_test)
    response = get_performances(description, responses_prediction, y_test)
    response = response + performances_cv
    return response, responses_prediction

#define the type of metrics
scoring = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

k_fold_value = 10

df = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

response = df['class']
df_data = df.drop(columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(df_data, response, test_size=0.3, random_state=42)

print("Exploring SVC")
clf = SVC()


response, responses_prediction = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "SVC", keys)

matrix_data = []
matrix_data.append(response)

df_export = pd.DataFrame(matrix_data, columns=['description', 'test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'fit_time', 'score_time', 'train_f1_weighted', 'train_recall_weighted', 'train_precision_weighted', 'train_accuracy'])
df_export.to_csv(path_export+"performances.csv", index=False)

# confusion matrix
cfm = confusion_matrix(y_test, responses_prediction)

df_matrix = pd.DataFrame(cfm, columns=['cerato_platanin', 'hydrophobin_class_I', 'hydrophobin_class_II'])
df_matrix.index = ['cerato_platanin', 'hydrophobin_class_I', 'hydrophobin_class_II']

df_matrix.to_csv(path_export+"confusion_matrix.csv")
