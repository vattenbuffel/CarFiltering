import torch
from state_transition import StateTransition
import pickle
import os 
import sys
sys.path.append(os.path.join(sys.path[0], '../../'))
from common_functions import map_angle
import numpy as np

class MyIterAbleDataSet(torch.utils.data.IterableDataset):
    def __init__(self, fp):
        super(MyIterAbleDataSet).__init__()
        self.data = [StateTransition]
        with open(fp, 'rb') as f:
            self.data = pickle.load(f)
        

    def __iter__(self):
        assert torch.utils.data.get_worker_info().num_workers == 1,  "Only allow single-process data loading"

        nn_xs = []
        nn_ys = []

        for trans in self.data:
            x_old = trans.x_old + trans.noise_old
            x_old[2:] = map_angle(x_old[2:])

            x_new = trans.x_new + trans.noise_new
            x_new[2:] = map_angle(x_new[2:])

            nn_x = np.vstack((x_old, trans.u, x_new))

            nn_xs.append(nn_x)
            nn_ys.append(trans.x_new)

        return iter([(nn_xs[i], nn_ys[i]) for i in range(len(self.data))])