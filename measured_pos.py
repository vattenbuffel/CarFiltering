import pygame
from common_functions import *

# not a good name
class MeasuredPos:
    def __init__(self, x, x_noise, u1u2, true_pos_color, noisy_pos_color) -> None:
        self.r = 4

        self.x = x
        self.x_noise = x_noise
        self.u1u2 = u1u2
        self.noisy_pos_color = noisy_pos_color
        self.true_pos_color = true_pos_color


    def true_pos_get(self):
        return self.x

    def noisy_pos_get(self):
        return self.x + self.x_noise
