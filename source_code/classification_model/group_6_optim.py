import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy
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

# Average CV score on the training set was: 0.8927493940402073
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            ),
            StandardScaler()
        ),
        FunctionTransformer(copy)
    ),
    RobustScaler(),
    MLPClassifier(alpha=0.1, learning_rate_init=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

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

print(dict_performances)

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