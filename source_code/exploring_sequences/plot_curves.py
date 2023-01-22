import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 14})

def create_df(data, index):

    array_data = []

    for column in data.columns:
        if "p_" in column:
            array_data.append(data[column][index])
    
    return array_data


full_df = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

groups = {
    "Group_0" : "Alpha structure",
    "Group_1" : "Beta structure",
    "Group_3" : "Volume",
    "Group_4" : "Energy",
    "Group_5" : "Hydropathy",
    "Group_7" : "Other indexes",
    "Group_2" : "Hydrophobicity",
    "Group_6" : "Secondary structure"
}

T = 1.0 / float(512)
xf = np.linspace(0.0, 1.0 / (2.0 * T), 512 // 2)

fig, axes = plt.subplots(4, 2, sharex=True, figsize=(16,18))

x_value = 0
y_value = 0
iteration = 1

for group in groups:
    print("Group ", groups[group])

    df_filter = full_df.loc[(full_df['Property'] == groups[group]) & (full_df['statistic'] == 'mean')]   
    
    df_filter = df_filter.reset_index()
    df_filter = df_filter.drop(columns=['statistic', 'index'])

    list_df = []

    for index in df_filter.index:
        array_data = create_df(df_filter, index)

        tmp_df = pd.DataFrame()
        tmp_df['Complex Modulus'] = array_data[30:]
        tmp_df['Type Protein'] = df_filter['Type Protein'][index]
        tmp_df['Frequency'] = xf[30:]
        
        list_df.append(tmp_df)
    
    df_to_plot = pd.concat(list_df, axis=0)

    y_label = r'$Complex\ modulus\ |F(\omega)|$'
    df_to_plot[y_label] = df_to_plot['Complex Modulus']

    g = sns.lineplot(ax=axes[x_value][y_value], x = "Frequency", y = y_label, data=df_to_plot, hue="Type Protein")
    
    axes[x_value][y_value].set_title(groups[group])

    if iteration != 8:
        axes[x_value][y_value].get_legend().remove()

    if iteration in [2, 4, 6]:
        x_value += 1
        y_value = 0
    else:
        y_value += 1
    
    iteration += 1

plt.savefig("{}summary_plots.pdf".format(path_export))