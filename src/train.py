import os
import pickle
import sys

import torch
from torch import nn
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.autograd import Variable
import pandas as pd
import numpy as np
import yaml
from torch import optim
import matplotlib.pyplot as plt
from model import my_model
from dataloader import set_up_data

params = yaml.safe_load(open("params.yaml"))["train"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

input_features = sys.argv[1]
output_model = sys.argv[2]
seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]

train_input = os.path.join(input_features, "train.csv")
test_input = os.path.join(input_features, "test.csv")


def get_df(filepath):
    df = pd.read_csv(filepath, encoding="utf-8", header=0, delimiter=",")
    sys.stderr.write(f"The input data frame {data} size is {df.shape}\n")
    return df


df_train = get_df(train_input)

train_data = set_up_data(df_train)
train_len = int(len(train_data) * 0.8)
test_len = len(train_data) - train_len
lengths = [train_len, test_len]
train_set, eval_set = torch.utils.data.random_split(
    train_data, lengths, generator=torch.Generator().manual_seed(42)
)
train_loader = data.DataLoader(train_set, batch_size=200, num_workers=0)
model = my_model()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.BCELoss()

loss_data = []

for e in range(n_est):
    running_loss = 0  # get images
    for info, labels in train_loader:
        # Flatten images
        # images=images.view(images.shape[0], -1)
        dummy_labels = labels.float()
        # info=info.reshape(info.shape[0],)
        dummy_labels = dummy_labels.view(-1, 1)
        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()

        # Forward pass
        output = model(info)

        # Calculate the loss
        loss = criterion(output, dummy_labels)

        # backward propagation
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    ## TODO: Implement the validation pass and print out the validation accuracy
    else:
        # print(f"Training loss: {running_loss/len(train_loader)}")
        loss_data.append(loss)
        # print('bias',model.linearlinear[0].bias)
        # print('weight',model.linearlinear[0].weight)
        # print("output",output)
        # print("labels",dummy_labels)
        # print('shape',info.shape)
# print("output",output)

epoch_count = range(1, n_est + 1)

# plt.plot(epoch_count, loss_data, "r")
# plt.legend(["Training Loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

print(f"Training loss: {running_loss/len(train_loader)}")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), output_model)
