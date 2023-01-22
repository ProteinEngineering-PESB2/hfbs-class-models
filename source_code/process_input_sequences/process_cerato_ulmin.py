import sys
from Bio import SeqIO
import pandas as pd

matrix_data = []

with open(sys.argv[1]) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        id_seq = record.id
        description = record.description.upper()
        sequence = str(record.seq)

        if "CERATO" in description:
            row = [id_seq, description, sequence]
            matrix_data.append(row)

df_data = pd.DataFrame(matrix_data, columns=['id', 'description', 'sequence'])
print(df_data)

df_data.to_csv(sys.argv[2], index=False)