class Filter:
    def __init__(self, x):
        self.x = x

    def update(self, x_measurement, u1, u2):
        raise NotImplementedError()