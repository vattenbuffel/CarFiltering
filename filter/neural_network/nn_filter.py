from filter.filter import Filter
import numpy as np
from filter.neural_network.network import MyNetwork
import torch

class NNFilter(Filter):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

        self.network = MyNetwork()
        self.network.load_state_dict(torch.load("filter/neural_network/network", map_location=torch.device('cpu')))

    def update(self, x_measurement, u1, u2):
        nn_x = torch.tensor(np.vstack((self.x, np.array((u1,u2)).reshape(-1,1), x_measurement)))
        self.x = self.network(nn_x).detach().numpy().reshape(-1,1)
        return self.x