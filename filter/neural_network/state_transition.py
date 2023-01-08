class StateTransition:
    def __init__(self, x_old, u, x_new, noise):
        self.x_old = x_old
        self.u = u
        self.x_new = x_new
        self.noise = noise