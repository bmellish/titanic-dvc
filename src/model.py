from torch import nn


class my_model(nn.Module):
    def __init__(self, n_in=12, n_hid=6, n_out=1):
        super(my_model, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out

        # self.fc1=nn.Linear(n_in,n_hid)

        self.linearlinear = nn.Sequential(
            nn.Linear(n_in, n_hid), nn.ReLU(), nn.Linear(n_hid, n_out), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linearlinear(x)

        return x
