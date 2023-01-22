import pandas as pd
import sys

df_input = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

df_process = df_input[['id','description','sequence']]

length_column = []

for index in df_process.index:
    length_column.append(len(df_process['sequence'][index]))

df_process['length'] = length_column

df_filter = df_process.loc[(df_process['length'] >=50) & (df_process['length'] <=512)]

name_export = "{}filter_sequences.csv".format(path_export)
df_filter.to_csv(name_export, index=False)