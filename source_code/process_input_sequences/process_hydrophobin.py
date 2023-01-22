import pandas as pd
import sys
from Bio import SeqIO
import re

#funcion que permite contar el numero de cys, debe ser mayor o igual a 8
def countNumberCys(sequence):

    contCys=0

    for i in range(len(sequence)):
        if sequence[i] == 'C':
            contCys+=1
    if contCys>=8:#cumple con el minimo
        return 0
    else:#no lo cumple
        return 1

#funcion que permite contar los cys-cys, debe ser mayor o igual a dos
def countNumberCysCys(sequence):

    contCysCys = 0

    for i in range(len(sequence)-1):
        if sequence[i] == 'C' and sequence[i+1] == 'C':
            contCysCys+=1
    if contCysCys>=2:#cumple con el minimo
        return 0
    else:#no lo cumple
        return 1
        
#pattern class II
def evaluatedPattern_class_II(sequence):

    patternClassI = re.compile('C[A-Z]{9,10}CC[A-Z]{11}C[A-Z]{16}C[A-Z]{8,9}CC[A-Z]{10}C')
    responseClass = patternClassI.findall(str(sequence))

    if len(responseClass)>0:
        return 0
    else:
        return 1

#pattern class I
def evaluatedPattern_class_I(sequence):

    patternClassI = re.compile('C[A-Z]{5,7}CC[A-Z]{19,39}C[A-Z]{8,23}C[A-Z]{5}CC[A-Z]{6,18}C')
    responseClass = patternClassI.findall(str(sequence))

    if len(responseClass)>0:
        return 0
    else:
        return 1

matrix_data = []

with open(sys.argv[1]) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        id_seq = record.id
        description = record.description.upper()
        sequence = str(record.seq)

        row = [id_seq, description, sequence]

        if "HYDROPHOBIN" in description:
            row.append(1)
        else:
            row.append(0)
        
        row.append(countNumberCys(sequence))
        row.append(countNumberCysCys(sequence))

        row.append(evaluatedPattern_class_II(sequence))
        row.append(evaluatedPattern_class_I(sequence))

        
        matrix_data.append(row)
df_data = pd.DataFrame(matrix_data, columns=['id', 'description', 'sequence', 'hydrophobin_in_number', 'number_cys', 'number_cys_cys', 'pattern_class_II', 'pattern_class_I'])


df_data_class_one = df_data.loc[df_data['pattern_class_I'] == 0]
df_data_class_two = df_data.loc[df_data['pattern_class_II'] == 0]

df_data_class_unknown = df_data.loc[(df_data['pattern_class_II'] == 1) & (df_data['pattern_class_I'] == 1)]

df_data_class_one.to_csv("class_I.csv", index=False)
df_data_class_two.to_csv("class_II.csv", index=False)
df_data_class_unknown.to_csv("unknown.csv", index=False)

print("class I: ", len(df_data_class_one))
print("class II: ", len(df_data_class_two))
print("class no idea: ", len(df_data_class_unknown))

