import pandas as pd
import sys
from joblib import load

import random

path_input = sys.argv[1]

list_df = []
list_models = []

responses = None

print("Loading models and dataframes")
for i in range(8):
    name_df = "{}Group_{}/encoding_data.csv".format(path_input, i)
    df_read = pd.read_csv(name_df)

    responses = df_read['class']
    features = df_read.drop('class', axis=1)

    list_df.append(features)

    name_model = "{}Group_{}/instance_model.joblib".format(path_input, i)
    model_instance = load(name_model)
    list_models.append(model_instance)

print("Apply the models")
df_responses = pd.DataFrame()
df_responses['real_values'] = responses

for i in range(8):
    print("Apply model ", i)

    responses = list_models[i].predict(list_df[i].values)
    name_column = "group_{}".format(i)

    df_responses[name_column] = responses

print("Get assembled response")
df_responses.to_csv("demo.csv", index=False)
df_responses_predict = df_responses.drop('real_values', axis=1)



    
        

