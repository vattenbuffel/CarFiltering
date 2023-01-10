import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from common_functions import load_config
import matplotlib.pyplot as plt
import pickle

from filter.low_pass.low_pass import LowPass
from filter.extended_kalman.extended_kalman import EKF
from filter.neural_network.nn_filter import NNFilter

config = load_config('config.yaml')

a = config['u1_factor']
b = config['u2_factor']
measurement_noise_std = config['measurement_noise_std']
true_pos_color = (0, 0, 0)
noisy_pos_color = (0, 0, 255)
filtered_pos_color = (0, 255, 0)

with open("filter/eval_data", 'rb') as f:
    data = pickle.load(f)
x_init = data[0].x_old

if config['filter'] == 'low_pass':
    filter = LowPass(x_init)
elif config['filter'] == 'EKF':
    filter = EKF(x_init)
elif config['filter'] == 'nn':
    filter = NNFilter(x_init)
else:
    raise NotImplementedError(f"No filter named: {config['filter']}")



xs = []
for d in data:
    xs.append(d.x_new)

plt.plt(range(len(xs)), xs)

