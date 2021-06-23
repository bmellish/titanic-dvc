import json
import math
import os
import pickle
import sys
import torch.utils.data as data
import torch
from model import my_model
import pandas as pd
from dataloader import set_up_data
import numpy as np

import sklearn.metrics as metrics

if len(sys.argv) != 6:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model features scores prc roc\n")
    sys.exit(1)

model_file = sys.argv[1]
matrix_file = os.path.join(sys.argv[2], "train.csv")
scores_file = sys.argv[3]
prc_file = sys.argv[4]
roc_file = sys.argv[5]


def get_df(data):
    df = pd.read_csv(data, encoding="utf-8", header=0, delimiter=",")
    sys.stderr.write(f"The input data frame {data} size is {df.shape}\n")
    return df


model = my_model()
model.load_state_dict(torch.load(model_file))
model.eval()
df = get_df(matrix_file)


eval_data = set_up_data(df)
train_len = int(len(eval_data) * 0.8)
test_len = len(eval_data) - train_len
lengths = [train_len, test_len]
train_set, eval_set = torch.utils.data.random_split(
    eval_data, lengths, generator=torch.Generator().manual_seed(42)
)
test_loader = data.DataLoader(eval_set, batch_size=200, num_workers=0)

predictions = np.array([[]])
labels = np.array([[]])

with torch.no_grad():
    for info, label in test_loader:
        output = model(info)
        labels = np.append(labels, label).astype(int)
        for i in range(len(info)):
            if output[i] > 0.5:
                predictions = np.append(predictions, "1").astype(int)

            else:
                predictions = np.append(predictions, "0").astype(int)

precision, recall, prc_thresholds = metrics.precision_recall_curve(labels, predictions)
fpr, tpr, roc_thresholds = metrics.roc_curve(labels, predictions)
avg_prec = metrics.average_precision_score(labels, predictions)
roc_auc = metrics.roc_auc_score(labels, predictions)

with open(scores_file, "w") as fd:
    json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

# ROC has a drop_intermediate arg that reduces the number of points.
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve.
# PRC lacks this arg, so we manually reduce to 1000 points as a rough estimate.
nth_point = math.ceil(len(prc_thresholds) / 1000)
prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
with open(prc_file, "w") as fd:
    json.dump(
        {
            "prc": [
                {"precision": float(p), "recall": float(r), "threshold": float(t)}
                for p, r, t in prc_points
            ]
        },
        fd,
        indent=4,
    )

with open(roc_file, "w") as fd:
    json.dump(
        {
            "roc": [
                {"fpr": float(fp), "tpr": float(tp), "threshold": float(t)}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        },
        fd,
        indent=4,
    )
