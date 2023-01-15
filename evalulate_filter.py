import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from common_functions import load_config
import matplotlib.pyplot as plt
import pickle
import numpy as np

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
data.pop(0)
x_init = data[0].x_old

if config['filter'] == 'low_pass':
    filter = LowPass(x_init)
elif config['filter'] == 'EKF':
    filter = EKF(x_init)
elif config['filter'] == 'nn':
    filter = NNFilter(x_init)
else:
    raise NotImplementedError(f"No filter named: {config['filter']}")



xs_true = []
xs_filtered = []
xs_measured = []
for d in data:
    xs_true.append(d.x_new)

    x_meas = d.x_new + d.noise_new
    xs_measured.append(x_meas)

    x_filter = filter.update(x_meas, *d.u)
    xs_filtered.append(x_filter)

xs_true = np.array(xs_true)
xs_measured = np.array(xs_measured)
xs_filtered = np.array(xs_filtered)

plt.plot(range(len(xs_true)), xs_true[:,0], label='True')
plt.plot(range(len(xs_measured)), xs_measured[:,0], label='Measured')
plt.plot(range(len(xs_filtered)), xs_filtered[:,0], label='Filtered')
plt.title('x')
plt.legend()
plt.show()

