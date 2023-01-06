from filter import Filter
from common_functions import load_config
import numpy as np
from numpy import sin, cos, tan

class EKF(Filter):
    def __init__(self, x):
        super().__init__(x)
        self.P = np.zeros((4,4))

        config = load_config('./extended_kalman/config.yaml')
        self.process_noise_std = config['process_noise_std']
        self.measurement_noise_std = config['measurement_noise_std']

        config = load_config('./config.yaml')
        self.a = config['u1_factor']
        self.b = np.deg2rad(config['u2_factor'])
        self.dt = config['dt']
        self.L = config['car_l']

    def F(self, u1, u2):
        res =  np.array([ 
             [1, 0, 0, 0], 
             [0, 1, 0, 0], 
             [0, 0, 1, 0], 
             [0, 0, 0, 1]], dtype='float')

        theta = self.x[2][0]

        res += np.array([ 
             [0, 0, -sin(theta)*self.a*u1, 0], 
             [0, 0, cos(theta)*self.b*u1, 0], 
             [0, 0, 0, 0], 
             [0, 0, 0, 0]]) * self.dt

        return res

    def model(self, u1, u2):
        theta = self.x[2, 0]
        L = self.L
        dt = self.dt

        v = np.array([
            cos(theta) * u1,
            sin(theta) * u1,
            1 / L * tan(u2) * u1,
            0,
        ]).reshape(4, 1)

        x1 = self.x + v * dt
        x1[3] = u2

        return x1


    def predict(self, u1, u2):
        self.x = self.model(self.a*u1, self.b*u2)
        F = self.F(self.a*u1, self.b*u2)
        self.P = F @ self.P @ F.transpose() + np.diag(np.full(4, self.process_noise_std))


    def correct(self, x_measurement):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        y = x_measurement - H @ self.x

        S = H @ self.P @ H.transpose() + np.diag(np.full(4, self.measurement_noise_std))
        K = self.P @ H.transpose() @ np.linalg.pinv(S)
        self.x = self.x + K@y
        self.P = (np.diag(np.full(4, 1)) - K@H) @ self.P


    def update(self, x_measurement, u1, u2):
        self.predict(u1, u2)
        self.correct(x_measurement)
        return self.x