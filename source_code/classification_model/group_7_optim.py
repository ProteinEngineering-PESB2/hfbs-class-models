import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive
import sys
import json
from joblib import dump

#for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

print("Reading dataset")
tpot_data = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

features = tpot_data.drop('class', axis=1)

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['class'])

print("Training model")
# Average CV score on the training set was: 0.8960242700758837
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=17, p=2, weights="uniform")),
    ZeroCount(),
    StackingEstimator(estimator=MultinomialNB(alpha=0.001, fit_prior=True)),
    MinMaxScaler(),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=0.05, min_samples_leaf=19, min_samples_split=3, n_estimators=100, subsample=0.7000000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)

print("Make predictions")
results = exported_pipeline.predict(testing_features)

print("Get performances")
accuracy_value = accuracy_score(testing_target, results)
f1_score_value = f1_score(testing_target, results, average='weighted')
precision_values = precision_score(testing_target, results, average='weighted')
recall_values = recall_score(testing_target, results, average='weighted')

dict_performances = {
    "accuracy_value" : accuracy_value,
    "f1_score_value" : f1_score_value,
    "precision_values" : precision_values,
    "recall_values" : recall_values
}

name_export = "{}summary_performances.json".format(path_export)
with open(name_export, 'w') as doc_export:
    json.dump(dict_performances, doc_export)

cfm = confusion_matrix(testing_target, results)

df_matrix = pd.DataFrame(cfm, columns=['cerato_platanin', 'hydrophobin_class_I', 'hydrophobin_class_II'])
df_matrix.index = ['cerato_platanin', 'hydrophobin_class_I', 'hydrophobin_class_II']

name_export = "{}confusion_matrix.csv".format(path_export)
df_matrix.to_csv(name_export)

name_export = "{}instance_model.joblib".format(path_export)
dump(exported_pipeline, name_export)