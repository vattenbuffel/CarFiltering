from filter import Filter
class LowPass(Filter):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

        self.alpha = 0.9
        self.beta = 1 - self.alpha

    def update(self, x_measurement, u1, u2):
        self.x = self.x*self.alpha + x_measurement*self.beta

        return self.x