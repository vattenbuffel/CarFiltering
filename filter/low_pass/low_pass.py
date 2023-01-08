from filter.filter import Filter
from common_functions import load_config

class LowPass(Filter):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

        self.config = load_config('./filter/low_pass/config.yaml')
        self.alpha = self.config['alpha']
        self.beta = 1 - self.alpha

    def update(self, x_measurement, u1, u2):
        self.x = self.x*self.alpha + x_measurement*self.beta

        return self.x