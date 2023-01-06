import numpy as np
from numpy import cos, sin, tan
from polygon_math import *
from common_functions import *
import pygame


class Car:

    def __init__(self) -> None:
        # todo: Wrap these up in a config
        self.L = 80
        self.dt = 0.5
        self.height = 50
        self.width = self.L

        self.x = np.zeros([4, 1])
        self.u1 = 0
        self.u2 = 0
        self.car_xs, self.car_ys = self.get_car_corners(self.x)
        self.car_xs.append(self.car_xs[0]), self.car_ys.append( self.car_ys[0])  # Make the polygon closed

    def step(self, u1u2, map_xs, map_ys):
        # u1u2 is a vector of u1 and u2. u1 and u2 are values between -1 and 1 giving the percentage value between min and max
        if isinstance(u1u2, np.ndarray):  #remove?
            u1u2 = u1u2.reshape(-1)

        self.u1, self.u2 = u1u2[0] * 5, u1u2[1] * np.deg2rad(30)  # Change 5 and 30 to be params

        # Update car state and check for collision
        x_temp = self.model(self.u1, self.u2, self.dt, self.L)
        car_xs_temp, car_ys_temp = self.get_car_corners(x_temp)
        car_xs_temp.append(car_xs_temp[0]), car_ys_temp.append(car_ys_temp[0])  # Make the polygon closed
        if ([], []) == get_polygon_intersection_points(car_xs_temp, car_ys_temp, map_xs, map_ys):
            self.x = x_temp
            self.car_xs, self.car_ys = car_xs_temp, car_ys_temp

    # @numba.njit
    def model(self, u1, u2, dt, L):
        # x is state vector [x, y, theta, phi]
        # u1 is vel, u2 is steering ang
        theta = self.x[2, 0]

        v = np.array([
            cos(theta) * u1,
            sin(theta) * u1,
            1 / L * tan(u2) * u1,
            0,
        ]).reshape(4, 1)

        x1 = self.x + v * dt
        x1[3] = u2

        return x1

    # @numba.njit
    def get_car_corners(self, x):
        x, y, theta = x[0][0], x[1][0], x[2][0]
        h = self.height
        w = self.width
        theta0 = np.arctan2(h / 2, w)  #angle at which p1 starts at
        d = ((h / 2)**2 + (w)**2)**0.5
        p0 = (h / 2 * cos(theta + np.pi / 2), h / 2 * sin(theta + np.pi / 2))
        p1 = (d * cos(theta + theta0), d * sin(theta + theta0))
        p2 = (d * cos(theta - theta0), d * sin(theta - theta0))
        p3 = (h / 2 * cos(theta - np.pi / 2), h / 2 * sin(theta - np.pi / 2))

        return [p0[0] + x, p1[0] + x, p2[0] + x, p3[0] + x], [p0[1] + y, p1[1] + y, p2[1] + y, p3[1] + y]

    def render(self, screen, screen_width, screen_height):
        p0 = (self.car_xs[0], self.car_ys[0])
        p1 = (self.car_xs[1], self.car_ys[1])
        p2 = (self.car_xs[2], self.car_ys[2])
        p3 = (self.car_xs[3], self.car_ys[3])

        p0 = pos_to_pix(*p0, screen_width, screen_height)
        p1 = pos_to_pix(*p1, screen_width, screen_height)
        p2 = pos_to_pix(*p2, screen_width, screen_height)
        p3 = pos_to_pix(*p3, screen_width, screen_height)

        pygame.draw.polygon(screen, BLUE, (p0, p1, p2, p3))
