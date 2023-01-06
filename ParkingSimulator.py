import numpy as np
import pygame
import time
from rolling_average import RollingAverage
import numba
from common_functions import GREEN, BLACK, WHITE, BLUE, pos_to_pix
from Wheel import Wheel
from polygon_math import *
from car import Car


class ParkingSimulator:

    def __init__(self) -> None:
        self.width, self.height = (720, 720)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.font.init()
        self.my_font = pygame.font.SysFont(None, 30)
        self.fps = RollingAverage(100)
        self.t_start = time.time()
        self.u1, self.u2 = 0, 0

        self.car = Car()
        wheel_width = 30
        wheel_height = 10
        self.wheel1 = Wheel(wheel_width, wheel_height,
                            self.car.width - wheel_width / 2 - 5,
                            -self.car.height / 2 + wheel_height / 2 + 5)
        self.wheel2 = Wheel(wheel_width, wheel_height,
                            self.car.width - wheel_width / 2 - 5,
                            self.car.height / 2 - wheel_height / 2 - 5)
        self.wheel3 = Wheel(wheel_width, wheel_height, wheel_width / 2 + 5,
                            self.car.height / 2 - wheel_height / 2 - 5)
        self.wheel4 = Wheel(wheel_width, wheel_height, wheel_width / 2 + 5,
                            -self.car.height / 2 + wheel_height / 2 + 5)

        self.map_xs = [-350, 350, 350, 50, 50, -100, -100, -350, -350]
        self.map_ys = [-350, -350, 350, 350, 100, 100, 350, 350, -350]

        self.reset()

    def reset(self):
        # TODO Make it spawn at random locations. To do this restructure obstacles and map to be convex
        self.car.x = np.zeros((4, 1))

        # Do this better...
        # self.x[0,0] = np.random.randint(-100, 100)
        # self.x[1,0] = np.random.randint(-100, 100)
        # self.x[2,0] = np.random.randint(0, 360)
        # while True:
        #     self.x[0,0] = np.random.randint(-self.width/2, self.width/2)
        #     self.x[1,0] = np.random.randint(-self.height/2, self.height/2)
        #     self.x[2,0] = np.random.randint(0, 360)

        #     xs, ys = get_car_corners(self.x[0,0], self.x[1, 0], self.x[3, 0], self.car.width, self.car.height)

        #     if valid_position(xs, ys, self.map_xs, self.map_ys):
        #         break

    def step(self, u1u2):
        self.t_start = time.time()
        self.u1, self.u2 = u1u2
        self.car.step(u1u2, self.map_xs, self.map_ys)

    def render(self):
        self.screen.fill(WHITE)

        x_ = self.car.x[0, 0]
        y = self.car.x[1, 0]
        theta = self.car.x[2, 0]
        phi = self.car.x[3, 0]

        # Draw stuff
        self.car.render(self.screen, self.width, self.height)
        self.wheel1.draw(x_, y, theta, phi, self.screen)
        self.wheel2.draw(x_, y, theta, phi, self.screen)
        self.wheel3.draw(x_, y, theta, 0, self.screen)
        self.wheel4.draw(x_, y, theta, 0, self.screen)
        draw_lines(self.map_xs, self.map_ys, self.screen, self.width, self.height)

        t_end = time.time()
        self.fps.update(1 / (t_end - self.t_start))
        str_ = f"u1: {self.u1:.2f}, u2: {np.rad2deg(self.u2):.2f}, fps: {self.fps.get():.2f}"
        text_surface = self.my_font.render(str_, True, BLACK)
        self.screen.blit(text_surface, (50, 50))

        pygame.display.update()



def draw_line(x0, y0, x1, y1, screen, screen_width, screen_height):
    p0 = pos_to_pix(x0, y0, screen_width, screen_height)
    p1 = pos_to_pix(x1, y1, screen_width, screen_height)
    pygame.draw.line(screen, BLACK, p0, p1)


def draw_lines(xs, ys, screen, screen_width, screen_height):
    for i in range(len(xs) - 1):
        draw_line(xs[i], ys[i], xs[i + 1], ys[i + 1], screen, screen_width, screen_height)


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
                elif events.dict['unicode'] == '\x1b':  # esc
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
