from filter import Filter
import yaml
from common_functions import load_config

class EKF(Filter):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

        self.config = load_config('./extended_kalman/config.yaml')

    def update(self, x_measurement, u1, u2):
        return self.x