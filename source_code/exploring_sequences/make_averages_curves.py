import pandas as pd
import sys
import numpy as np

path_to_process = sys.argv[1]

groups = ['Group_{}'.format(i) for i in range(8)]

for group in groups:
    print("Processing group ", group)
    df_data = pd.read_csv("{}{}/fft_property_encoder.csv".format(path_to_process, group))
    df_summary = df_data.describe()
    df_summary.to_csv("{}{}/statistic_summary.csv".format(path_to_process, group))