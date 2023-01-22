import pandas as pd
import sys
import numpy as np

df = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

len_array = []

for i in range(len(df)):
    len_array.append(len(df['sequence'][i]))

df_export = pd.DataFrame()

df_export['id'] = df['id']
df_export['length_seq'] = len_array

average = np.mean(len_array)
std = np.std(len_array)

filter_data = average + (1.5*std)

print(filter_data)
filter_by_length = []

for i in range(len(df_export)):
    if df_export['length_seq'][i] < 50 or df_export['length_seq'][i] > filter_data:
        filter_by_length.append(1)
    else:
        filter_by_length.append(0)

df_export['filter'] = filter_by_length

df_merge = df_export.merge(df, left_on='id', right_on='id')

filter_to_encoder = df_merge.loc[df_merge['filter'] == 0]

print(len(df))
print(len(filter_to_encoder))

filter_to_encoder.to_csv("{}filter_seqs.csv".format(path_export), index=False)