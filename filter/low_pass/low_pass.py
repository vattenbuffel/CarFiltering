try:
    from filter.filter import Filter
except ModuleNotFoundError:
    from filter import Filter
    
from common_functions import load_config, map_angle

class LowPass(Filter):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

        self.config = load_config('./filter/low_pass/config.yaml')
        self.alpha = self.config['alpha']
        self.beta = 1 - self.alpha

    def update(self, x_measurement, u1, u2):
        self.x = self.x*self.alpha + x_measurement*self.beta
        self.x[2:] = map_angle(self.x[2:])

        return self.x