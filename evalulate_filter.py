import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from common_functions import load_config, map_angle
import matplotlib.pyplot as plt
import pickle
import numpy as np

from filter.low_pass.low_pass import LowPass
from filter.extended_kalman.extended_kalman import EKF
from filter.neural_network.nn_filter import NNFilter

PLOT_EVERY = 10

config = load_config('config.yaml')

dt = config['dt']
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



xs_true = []
xs_filtered = []
xs_measured = []
for d in data:
    xs_true.append(d.x_new)

    x_meas = d.x_new + d.noise_new
    xs_measured.append(x_meas)

    x_filter = filter.update(x_meas, *d.u)
    xs_filtered.append(x_filter)

xs_true = np.array(xs_true)[::PLOT_EVERY]
xs_measured = np.array(xs_measured)[::PLOT_EVERY]
xs_filtered = np.array(xs_filtered)[::PLOT_EVERY]
t = np.linspace(0, len(xs_true)*dt, len(xs_true))

plt.figure(1)
plt.scatter(t, xs_true[:,0], label='True', marker='o', alpha=0.6)
plt.scatter(t, xs_measured[:,0], label='Measured', marker='x', alpha=0.6)
plt.scatter(t, xs_filtered[:,0], label='Filtered', marker=',', alpha=0.6)
plt.title('x')
plt.legend()

plt.figure(2)
plt.scatter(t, xs_true[:,1], label='True', marker='o', alpha=0.6)
plt.scatter(t, xs_measured[:,1], label='Measured', marker='x', alpha=0.6)
plt.scatter(t, xs_filtered[:,1], label='Filtered', marker=',', alpha=0.6)
plt.title('y')
plt.legend()

plt.figure(3)
plt.scatter(t, np.rad2deg(map_angle(xs_true[:,2])), label='True', marker='o', alpha=0.6)
plt.scatter(t, np.rad2deg(map_angle(xs_measured[:,2])), label='Measured', marker='x', alpha=0.6)
plt.scatter(t, np.rad2deg(map_angle(xs_filtered[:,2])), label='Filtered', marker=',', alpha=0.6)
plt.title('\theta')
plt.legend()

plt.figure(4)
plt.scatter(t, np.rad2deg(map_angle(xs_true[:,3])), label='True', marker='o', alpha=0.6)
plt.scatter(t, np.rad2deg(map_angle(xs_measured[:,3])), label='Measured', marker='x', alpha=0.6)
plt.scatter(t, np.rad2deg(map_angle(xs_filtered[:,3])), label='Filtered', marker=',', alpha=0.6)
plt.title('\phi')
plt.legend()


plt.show()

