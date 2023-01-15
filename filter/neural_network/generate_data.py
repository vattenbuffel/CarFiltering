import numpy as np
import os 
import sys
sys.path.append(os.path.join(sys.path[0], '../../'))
from state_transition import StateTransition
from car import Car
from common_functions import load_config, map_angle
import pickle



n_data = 5000
n_train = int(n_data*0.8)
n_val = n_data - n_train


config = load_config('config.yaml')


def random_pos(n_data):
    x_max = 1000
    x_min = -1000
    y_max = 1000
    y_min = -1000
    theta_max = np.pi
    theta_min = -np.pi
    phi_max = np.deg2rad(config['u2_factor'])
    phi_min = np.deg2rad(-config['u2_factor'])


    x = np.random.uniform(x_min, x_max, n_data)
    y = np.random.uniform(y_min, y_max, n_data)
    theta = np.random.uniform(theta_min, theta_max, n_data)
    phi = np.random.uniform(phi_min, phi_max, n_data)

    return np.array([x, y, theta, phi])

def get_noise(x):
    noise = np.random.normal(0, config['measurement_noise_std'], x.shape)
    noise[2:, :] = np.deg2rad(noise[2:, :])
    return noise

def random_u(n_data):
    u1_max = config['u1_factor']
    u1_min = -config['u1_factor']
    u2_max = np.deg2rad(config['u2_factor'])
    u2_min = np.deg2rad(-config['u2_factor'])

    u1 = np.random.uniform(u1_min, u1_max, n_data)
    u2 = np.random.uniform(u2_min, u2_max, n_data)

    return np.array([u1, u2])


car = Car()
train_pos = []
val_pos = []
dt = config['dt']
L = config['car_l']

xs = random_pos(n_train)
us = random_u(n_train)
old_noise = get_noise(xs)
new_noise = get_noise(xs)
for i in range(n_train):
    car.x = xs[:, i].reshape(-1, 1)
    u1, u2 = us[0, i], us[1, i]
    x_new = car.model(u1, u2, dt, L)
    x_new[2:] = map_angle(x_new[2:])

    state_transition = StateTransition(xs[:, i].reshape(-1,1), old_noise[:, i].reshape(-1, 1), us[:, 1].reshape(-1, 1), x_new, new_noise[:, i].reshape(-1,1))
    train_pos.append(state_transition)

for i in range(n_val):
    car.x = xs[:, i].reshape(-1, 1)
    u1, u2 = us[0, i], us[1, i]
    x_new = car.model(u1, u2, dt, L)

    state_transition = StateTransition(xs[:, i].reshape(-1,1), old_noise[:, i].reshape(-1, 1), us[:, 1].reshape(-1, 1), x_new, new_noise[:, i].reshape(-1,1))
    val_pos.append(state_transition)

with open("filter/neural_network/train_data", 'wb') as f:
    pickle.dump(train_pos, f)
with open("filter/neural_network/val_data", 'wb') as f:
    pickle.dump(train_pos, f)











