#for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import pandas as pd

df_responses_data = pd.read_csv("demo.csv")
df_responses_predict = df_responses_data.drop(columns=["real_values"])

values_response = []

for i in range(len(df_responses_predict)):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0

    for column in df_responses_predict.columns:
        if df_responses_predict[column][i] == 1:
            sum_1+=1
        if df_responses_predict[column][i] == 2:
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

df_responses_data['voted_prediction'] = values_response

print("Get performances")
accuracy_value = accuracy_score(df_responses_data['real_values'], df_responses_data['voted_prediction'])
f1_score_value = f1_score(df_responses_data['real_values'], df_responses_data['voted_prediction'], average='weighted')
precision_values = precision_score(df_responses_data['real_values'], df_responses_data['voted_prediction'], average='weighted')
recall_values = recall_score(df_responses_data['real_values'], df_responses_data['voted_prediction'], average='weighted')

dict_performances = {
    "accuracy_value" : accuracy_value,
    "f1_score_value" : f1_score_value,
    "precision_values" : precision_values,
    "recall_values" : recall_values
}

print(dict_performances)
