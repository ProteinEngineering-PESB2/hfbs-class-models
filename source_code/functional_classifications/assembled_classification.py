import pandas as pd
import sys
from joblib import load

path_input_models = sys.argv[1]
path_input_csv = sys.argv[2]

list_df = []
list_models = []

list_id = None

print("Loading models and dataframes")
for i in range(8):
    name_df = "{}Group_{}/fft_property_encoder.csv".format(path_input_csv, i)
    df_read = pd.read_csv(name_df)

    list_id = df_read['id']
    features = df_read.drop('id', axis=1)

    list_df.append(features)

    name_model = "{}Group_{}/instance_model.joblib".format(path_input_models, i)
    model_instance = load(name_model)
    list_models.append(model_instance)

print("Apply the models")
df_responses = pd.DataFrame()
df_responses['id'] = list_id

for i in range(8):
    print("Apply model ", i)

    responses = list_models[i].predict(list_df[i].values)
    name_column = "group_{}".format(i)

    df_responses[name_column] = responses

print("Get assembled response")
values_response = []

for i in range(len(df_responses)):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0

    for column in df_responses.columns:
        if column != "id":
            if df_responses[column][i] == 1:
                sum_1+=1
            if df_responses[column][i] == 2:
                sum_2+=1
            else:
                sum_3+=1
    
    row = [sum_1, sum_2, sum_3]

    max_value = max(row)

    if max_value == sum_1:
        prediction = 1
    elif max_value == sum_2:
        prediction = 2
    else:
        prediction = 3
    
    values_response.append(prediction)

df_responses['voted_prediction'] = values_response

print("Export responses")
name_export = "{}prediction_responses.csv".format(path_input_csv)
df_responses.to_csv(name_export, index=False)