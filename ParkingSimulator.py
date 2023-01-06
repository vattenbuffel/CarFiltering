import numpy as np
from numpy import cos, sin, tan
import pygame
import time
from rolling_average import RollingAverage
import numba
from common_functions import GREEN, BLACK, WHITE, BLUE, pos_to_pix
from Wheel import Wheel
from polygon_math import *


class ParkingSimulator:
    def __init__(self) -> None:
        self.width, self.height = (720, 720)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.font.init()
        self.my_font = pygame.font.SysFont(None, 30)
        self.fps = RollingAverage(100)
        self.t_start = time.time()
        self.draw_scan_lines = False


        self.x = np.zeros([4, 1])
        self.u1 = 0
        self.u2 = 0
        self.dt = 0.5
        self.L = 80
        self.car_width = self.L
        self.car_height = 50
        self.car_xs , self.car_ys = get_car_corners(self.x[0, 0], self.x[1, 0], self.x[2, 0], self.car_width, self.car_height)
        self.car_xs.append(self.car_xs[0]), self.car_ys.append(self.car_ys[0]) # Make the polygon closed
        self.reward = -1

        wheel_width = 30
        wheel_height = 10
        self.wheel1 = Wheel(wheel_width, wheel_height, self.car_width - wheel_width/2 - 5, -self.car_height/2 + wheel_height/2 + 5)
        self.wheel2 = Wheel(wheel_width, wheel_height, self.car_width - wheel_width/2 - 5, self.car_height/2 - wheel_height/2 - 5)
        self.wheel3 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, self.car_height/2 - wheel_height/2 - 5)
        self.wheel4 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, -self.car_height/2 + wheel_height/2 + 5)

        self.map_xs = [-350, 350, 350, 50, 50, -100, -100, -350, -350]
        self.map_ys  = [-350, -350, 350, 350, 100, 100, 350, 350, -350]

        self.reset()


    def reset(self):
        # TODO Make it spawn at random locations. To do this restructure obstacles and map to be convex
        self.x = np.zeros((4, 1))

        # Do this better...
        # self.x[0,0] = np.random.randint(-100, 100)
        # self.x[1,0] = np.random.randint(-100, 100)
        # self.x[2,0] = np.random.randint(0, 360)
        # while True:
        #     self.x[0,0] = np.random.randint(-self.width/2, self.width/2)
        #     self.x[1,0] = np.random.randint(-self.height/2, self.height/2)
        #     self.x[2,0] = np.random.randint(0, 360)

        #     xs, ys = get_car_corners(self.x[0,0], self.x[1, 0], self.x[3, 0], self.car_width, self.car_height)

        #     if valid_position(xs, ys, self.map_xs, self.map_ys):
        #         break




    def step(self, u1u2):
        # u1u2 is a vector of u1 and u2. u1 and u2 are values between -1 and 1 giving the percentage value between min and max
        self.t_start = time.time()
        if isinstance(u1u2, np.ndarray):
            u1u2 = u1u2.reshape(-1)

        self.u1, self.u2 = u1u2[0]*5, u1u2[1]*np.deg2rad(30) # Change 5 and 30 to be params

        # Update car state and check for collision
        x_temp = model(self.x, self.u1, self.u2, self.dt, self.L)
        car_xs_temp, car_ys_temp = get_car_corners(x_temp[0, 0], x_temp[1, 0], x_temp[2, 0], self.car_width, self.car_height)
        car_xs_temp.append(car_xs_temp[0]), car_ys_temp.append(car_ys_temp[0]) # Make the polygon closed
        if ([], []) == get_polygon_intersection_points(car_xs_temp, car_ys_temp, self.map_xs, self.map_ys):
            self.x = x_temp
            self.car_xs, self.car_ys = car_xs_temp, car_ys_temp



    def render(self):
        # consume events is needed to be done when training
        self.screen.fill(WHITE)

        x_ = self.x[0, 0]
        y = self.x[1, 0]
        theta = self.x[2, 0]
        phi = self.x[3, 0]


        # Draw stuff
        draw_car(self.car_xs[:-1], self.car_ys[:-1], self.screen, self.width, self.height)
        self.wheel1.draw(x_, y, theta, phi, self.screen)
        self.wheel2.draw(x_, y, theta, phi, self.screen)
        self.wheel3.draw(x_, y, theta, 0, self.screen)
        self.wheel4.draw(x_, y, theta, 0, self.screen)
        draw_lines(self.map_xs, self.map_ys, self.screen, self.width, self.height)

        t_end = time.time()
        self.fps.update(1/(t_end - self.t_start))
        str_ = f"u1: {self.u1:.2f}, u2: {np.rad2deg(self.u2):.2f}, fps: {self.fps.get():.2f}"
        text_surface = self.my_font.render(str_, True, BLACK)
        self.screen.blit(text_surface, (50,50))

        pygame.display.update()


@numba.njit
def valid_position(car_xs, car_ys, map_xs, map_ys):
    # this doesn't work as point_in_polygon needs the polygon to be convex and map is not
    for i in range(len(car_xs)):
        if not point_in_polygon(map_xs, map_ys, car_xs[i], car_ys[i]):
            return False

    return True


@numba.njit
def clip(val, min_, max_):
    return np.minimum(max_, np.maximum(min_, val))

@numba.njit
def model(x, u1, u2, dt, L):
    # x is state vector [x, y, theta, phi]
    # u1 is vel, u2 is steering ang
    u1 = clip(u1, -5, 5) # Change 5 to be param
    u2 = clip(u2, -np.deg2rad(30), np.deg2rad(30)) # Change 30 to be param

    theta = x[2, 0]

    v = np.array([
        cos(theta)*u1,
        sin(theta)*u1,
        1/L*tan(u2)*u1,
        0,
    ]).reshape(4, 1)

    x1 = x + v*dt
    x1[3] = u2

    return x1


@numba.njit
def get_car_corners(x, y, theta, car_width, car_height):
    h = car_height
    w = car_width
    theta0 = np.arctan2(h/2, w)   #angle at which p1 starts at
    d = ((h/2)**2 + (w)**2)**0.5
    p0 = (h/2*cos(theta + np.pi/2), h/2*sin(theta + np.pi/2))
    p1 = (d * cos(theta + theta0), d * sin(theta + theta0))
    p2 = (d * cos(theta - theta0), d * sin(theta - theta0))
    p3 = (h/2*cos(theta - np.pi/2), h/2*sin(theta - np.pi/2))

    return [p0[0]+x, p1[0]+x, p2[0]+x, p3[0]+x], [p0[1]+y, p1[1]+y, p2[1]+y, p3[1]+y]


def draw_car(car_xs, car_ys, screen, screen_width, screen_height):
    p0 = (car_xs[0], car_ys[0])
    p1 = (car_xs[1], car_ys[1])
    p2 = (car_xs[2], car_ys[2])
    p3 = (car_xs[3], car_ys[3])

    p0 = pos_to_pix(*p0, screen_width, screen_height)
    p1 = pos_to_pix(*p1, screen_width, screen_height)
    p2 = pos_to_pix(*p2, screen_width, screen_height)
    p3 = pos_to_pix(*p3, screen_width, screen_height)

    pygame.draw.polygon(screen, BLUE, (p0, p1, p2, p3))


def draw_line(x0, y0, x1, y1, screen, screen_width, screen_height):
    p0 = pos_to_pix(x0, y0, screen_width, screen_height)
    p1 = pos_to_pix(x1, y1, screen_width, screen_height)
    pygame.draw.line(screen, BLACK, p0, p1)

def draw_lines(xs, ys, screen, screen_width, screen_height):
    for i in range(len(xs)-1):
        draw_line(xs[i], ys[i], xs[i+1], ys[i+1], screen, screen_width, screen_height)



if __name__ == '__main__':
    env = ParkingSimulator()
    env.reset()
    u1, u2 = 0, 0

    while True:
        for events in pygame.event.get():
            if events.type == pygame.QUIT:
                import sys
                sys.exit(0)
            elif events.type == pygame.KEYDOWN:
                if events.dict['unicode'] == 'w':
                    u1 = 1
                elif events.dict['unicode'] == 'a':
                    u2 = 1
                elif events.dict['unicode'] == 'd':
                    u2 = -1
                elif events.dict['unicode'] == 's':
                    u1 = -1
                elif events.dict['unicode'] == '\x1b': # esc
                    exit(0)
                elif events.dict['unicode'] == '\x1b': # esc
                    exit(0)
                elif events.dict['unicode'] == ' ':
                    env.reset()
            elif events.type == pygame.KEYUP:
                if events.dict['unicode'] == 'w':
                    u1 = 0
                elif events.dict['unicode'] == 'a':
                    u2 = 0
                elif events.dict['unicode'] == 'd':
                    u2 = 0
                elif events.dict['unicode'] == 's':
                    u1 = 0

        env.step((u1, u2))
        env.render()

