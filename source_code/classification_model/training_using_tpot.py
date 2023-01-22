import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


df = pd.read_csv(sys.argv[1])
name_export = sys.argv[2]

response = df['class']
df_data = df.drop(columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(df_data, response, test_size=0.3, random_state=42)

tpot = TPOTClassifier(generations=50, population_size=50, verbosity=2, random_state=42, n_jobs=-1)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export(name_export)