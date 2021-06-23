import os
import pickle
import sys
import numpy as np
import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))["featurize"]

np.set_printoptions(suppress=True)

if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
    sys.exit(1)

train_input = os.path.join(sys.argv[1], "train.tsv")
test_input = os.path.join(sys.argv[1], "test.tsv")
train_output = os.path.join(sys.argv[2], "train.csv")
test_output = os.path.join(sys.argv[2], "test.csv")

max_features = params["max_features"]
ngrams = params["ngrams"]


def get_df(data):
    df = pd.read_csv(data, encoding="utf-8", header=0, delimiter=",")
    sys.stderr.write(f"The input data frame {data} size is {df.shape}\n")
    return df


def save_matrix(df, output):

    with open(output, "wb") as fd:
        df.to_csv(fd)
    pass


def fill_in_age(x):
    if x["Pclass_1"] == 1:
        return 37
    elif x["Pclass_2"] == 1:
        return 29
    else:
        return 25


os.makedirs(sys.argv[2], exist_ok=True)

# Generate train feature matrix
df_train = get_df(train_input)
# labels=titanic['Survived']
dummy_fields = ["Pclass", "Sex", "Embarked"]
for each in dummy_fields:
    dummies = pd.get_dummies(df_train[each], prefix=each, drop_first=False)
    df_train = pd.concat([df_train, dummies], axis=1)


fields_to_drop = ["PassengerId", "Cabin", "Pclass", "Name", "Sex", "Ticket", "Embarked"]
df = df_train.drop(fields_to_drop, axis=1)

df["Age"] = df.apply(fill_in_age, axis=1)

to_normalize = ["Age", "Fare"]
for each in to_normalize:
    mean, std = df[each].mean(), df[each].std()
    df.loc[:, each] = (df[each] - mean) / std


save_matrix(df, train_output)

# Generate test feature matrix
df_test = get_df(test_input)
dummy_fields = ["Pclass", "Sex", "Embarked"]
for each in dummy_fields:
    dummies = pd.get_dummies(df_test[each], prefix=each, drop_first=False)
    df_test = pd.concat([df_test, dummies], axis=1)


fields_to_drop = ["PassengerId", "Cabin", "Pclass", "Name", "Sex", "Ticket", "Embarked"]
df = df_test.drop(fields_to_drop, axis=1)

df["Age"] = df.apply(fill_in_age, axis=1)

to_normalize = ["Age", "Fare"]
for each in to_normalize:
    mean, std = df[each].mean(), df[each].std()
    df.loc[:, each] = (df[each] - mean) / std


save_matrix(df, test_output)
