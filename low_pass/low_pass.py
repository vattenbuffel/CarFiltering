from filter import Filter
class LowPass(Filter):
    def __init__(self, x):
        super().__init__(x)
        self.x = x