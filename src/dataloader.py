import torch
import torch.utils.data as data


class set_up_data(data.Dataset):
    def __init__(self, dataframe):
        data_matrix = dataframe.values
        data_matrix = torch.from_numpy(
            data_matrix
        )  # converting tensor to numpy for matrix operations
        self.data = data_matrix[:, 2:14]  # columns 2-14 are the data
        self.data = self.data.float()
        self.target = data_matrix[:, 1]  # column 1 is the true output

        self.n_samples = self.data.shape[0]

    def __len__(self):  # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):  # Function that returns one point and one label.
        # return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])
        return self.data[index], self.target[index]
