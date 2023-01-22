import pandas as pd
import sys

dir_process = {
    "cerato_platanin" : "Cerato platanin",
    "hydrophobin_class_I" : "Hydrophobin Class I",
    "hydrophobin_class_II" : "Hydrophobin Class II"
}

groups = {
    "Group_0" : "Alpha structure",
    "Group_1" : "Beta structure",
    "Group_2" : "Hydrophobicity",
    "Group_3" : "Volume",
    "Group_4" : "Energy",
    "Group_5" : "Hydropathy",
    "Group_6" : "Secondary structure",
    "Group_7" : "Other indexes"
}

path_to_process = sys.argv[1]

df_list = []

for path in dir_process:
    for group in groups:
        df_read = pd.read_csv("{}{}/{}/statistic_summary.csv".format(path_to_process, path, group))

        df_read['Property'] = groups[group]
        df_read['Type Protein'] = dir_process[path]
        df_list.append(df_read)

df_concat = pd.concat(df_list, axis=0)
df_concat.to_csv("{}full_data_summary.csv".format(path_to_process), index=False)



