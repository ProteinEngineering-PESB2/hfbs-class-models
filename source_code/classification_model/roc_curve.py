import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC

from itertools import combinations

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from itertools import cycle

target_names = ["Cerato platanin", "Hydrophobin Class I", "Hydrophobin Class II"]

print("Reading data")
df = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

array_response = []

for index in df.index:
    if df['class'][index] == 1:
        array_response.append(target_names[0])
    elif df['class'][index] == 2:
        array_response.append(target_names[1])
    else:
        array_response.append(target_names[2])

df['class'] = array_response

response = df['class']
df_data = df.drop(columns=['class'])

print("Prepare dataset")
random_state = np.random.RandomState(0)
n_samples, n_features = df_data.shape
n_classes = len(np.unique(response))
df_data = np.concatenate([df_data, random_state.randn(n_samples, 200 * n_features)], axis=1)

(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(df_data, response, test_size=0.5, stratify=response, random_state=0)


print("Training model")
clf = SVC(probability=True)
clf.fit(X_train, y_train)

print("Get probability")
y_score = clf.predict_proba(X_test)


label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

pair_list = list(combinations(np.unique(response), 2))
print(pair_list)

print("Start to plot")
pair_scores = []
mean_tpr = dict()

fpr_grid = np.linspace(0.0, 1.0, 1000)
for ix, (label_a, label_b) in enumerate(pair_list):

    a_mask = y_test == label_a
    b_mask = y_test == label_b
    ab_mask = np.logical_or(a_mask, b_mask)

    a_true = a_mask[ab_mask]
    b_true = b_mask[ab_mask]

    idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]
    idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]

    fpr_a, tpr_a, _ = roc_curve(a_true, y_score[ab_mask, idx_a])
    fpr_b, tpr_b, _ = roc_curve(b_true, y_score[ab_mask, idx_b])

    mean_tpr[ix] = np.zeros_like(fpr_grid)
    mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
    mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
    mean_tpr[ix] /= 2
    mean_score = auc(fpr_grid, mean_tpr[ix])
    pair_scores.append(mean_score)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(
        fpr_grid,
        mean_tpr[ix],
        label=f"Mean {label_a} vs {label_b} (AUC = {mean_score :.2f})",
        linestyle=":",
        linewidth=4,
    )
    RocCurveDisplay.from_predictions(
        a_true,
        y_score[ab_mask, idx_a],
        ax=ax,
        name=f"{label_a} as positive class",
    )
    RocCurveDisplay.from_predictions(
        b_true,
        y_score[ab_mask, idx_b],
        ax=ax,
        name=f"{label_b} as positive class",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{target_names[idx_a]} vs {label_b} ROC curves")
    plt.legend()
    plt.savefig("{}roc_curve_{}.png".format(path_export, ix))

    plt.clf()
