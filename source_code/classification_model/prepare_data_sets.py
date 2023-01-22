import pandas as pd
import sys
import os

path_input = sys.argv[1]
path_export = sys.argv[2]

groups = ['Group_{}'.format(i) for i in range(8)]

dir_process = {
    "cerato_platanin" : 1,
    "hydrophobin_class_I" : 2,
    "hydrophobin_class_II" : 3
}

for group in groups:
    print("Processing group :", group)
    list_df = []

    for element in dir_process:
        df_read = pd.read_csv("{}{}/{}/fft_property_encoder.csv".format(path_input, element, group))

        df_read['class'] = dir_process[element]
        list_df.append(df_read)
    
    df_process = pd.concat(list_df, axis=0)
    df_process = df_process.reset_index()
    df_process = df_process.drop(columns=['index', 'id'])
    
    command = "mkdir -p {}{}".format(path_export, group)
    print(command)
    os.system(command)

    name_export = "{}{}/encoding_data.csv".format(path_export, group)
    df_process.to_csv(name_export, index=False)
