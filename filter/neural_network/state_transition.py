class StateTransition:
    def __init__(self, x_old, noise_old, u, x_new, noise_new):
        # Theta in x is in rad
        self.x_old = x_old
        self.noise_old = noise_old # Used for training nn so it doesn't get the true state as input
        self.u = u
        self.x_new = x_new
        self.noise_new = noise_new

    def __str__(self):
        return f"x_old: {self.x_old}, noise: {self.noise}, u: {self.u}, x_new: {self.x_new}"