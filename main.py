import numpy as np
import pygame
import time
from rolling_average import RollingAverage
import numba
from common_functions import GREEN, BLACK, WHITE, BLUE, pos_to_pix, load_config
from Wheel import Wheel
from polygon_math import *
from car import Car
from low_pass.low_pass import LowPass
from extended_kalman.extended_kalman import EKF
from collections import deque
from measured_pos import MeasuredPos

class ParkingSimulator:

    def __init__(self) -> None:
        config = load_config('config.yaml')
        self.width, self.height = (config['screen_width'], config['screen_height'])
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.font.init()
        self.my_font = pygame.font.SysFont(None, 30)
        self.fps = RollingAverage(100)
        self.t_start = time.time()
        self.u1, self.u2 = 0, 0

        self.car = Car()
        wheel_width = config['wheel_width']
        wheel_height = config['wheel_height']
        self.wheel1 = Wheel(wheel_width, wheel_height, self.car.width - wheel_width / 2 - 5, -self.car.height / 2 + wheel_height / 2 + 5)
        self.wheel2 = Wheel(wheel_width, wheel_height, self.car.width - wheel_width / 2 - 5, self.car.height / 2 - wheel_height / 2 - 5)
        self.wheel3 = Wheel(wheel_width, wheel_height, wheel_width / 2 + 5, self.car.height / 2 - wheel_height / 2 - 5)
        self.wheel4 = Wheel(wheel_width, wheel_height, wheel_width / 2 + 5, -self.car.height / 2 + wheel_height / 2 + 5)

        if config['filter'] == 'low_pass':
            self.filter = LowPass(self.car.x)
        elif config['filter'] == 'EKF':
            self.filter = EKF(self.car.x)
        else:
            raise NotImplementedError(f"No filter named: {config['filter']}")


        self.prev_pos = deque(maxlen=100)
        self.filterd_pos = deque(maxlen=100)
        self.measurement_noise_std = config['measurement_noise_std']
        self.true_pos_color = (0, 0, 0)
        self.noisy_pos_color = (0, 0, 255)
        self.filtered_pos_color = (0, 255, 0)
        self.pos_save_counter = 0

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

        measurement_noise = np.random.normal(0, self.measurement_noise_std, (4,1))
        measured_pos = MeasuredPos(self.car.x, measurement_noise, u1u2, self.true_pos_color, self.noisy_pos_color)
        filtered_pos = self.filter.update(measured_pos.noisy_pos_get(), *u1u2)
        filtered_pos = MeasuredPos(filtered_pos, 0, u1u2, self.filtered_pos_color, self.filtered_pos_color)
        if self.pos_save_counter == 5:
            self.prev_pos.append(measured_pos)
            self.filterd_pos.append(filtered_pos)
            self.pos_save_counter = 0
        if u1u2 != (0,0):
            self.pos_save_counter += 1

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
        draw_pos(self.prev_pos, self.filterd_pos, self.screen, self.width, self.height)

        t_end = time.time()
        self.fps.update(1 / (t_end - self.t_start))
        noisy_pos_rms, filter_rms = rms(self.prev_pos, self.filterd_pos) if len(self.prev_pos) else (-1, -1)
        str_ = f"u1: {self.u1:.2f}, u2: {np.rad2deg(self.u2):.2f}, fps: {self.fps.get():.2f}, filter_rms: {filter_rms:.2f}, noisy_pos_rms: {noisy_pos_rms:.2f}"
        text_surface = self.my_font.render(str_, True, BLACK)
        self.screen.blit(text_surface, (50, 50))

        pygame.display.update()


def draw_pos(prev_pos: [MeasuredPos], filtered_pos: [MeasuredPos], screen, screen_width, screen_height):
    for p in prev_pos:
        x = p.true_pos_get()
        draw_circle(x[0][0], x[1][0], p.true_pos_color, p.r, screen, screen_width, screen_height)
        # draw_arrow(measured_pos.noisy_pos_get()[:3])

        x = p.noisy_pos_get()
        draw_circle(x[0][0], x[1][0], p.noisy_pos_color, p.r, screen, screen_width, screen_height)
        # draw_arrow(measured_pos.noisy_pos_get()[:3])

    for p in filtered_pos:
        x = p.true_pos_get()
        draw_circle(x[0][0], x[1][0], p.true_pos_color, p.r, screen, screen_width, screen_height)
        # draw_arrow(measured_pos.noisy_pos_get()[:3])



def draw_circle(x, y, color, r, screen, screen_width, screen_height):
    p = pos_to_pix(x, y, screen_width, screen_height)
    pygame.draw.circle(screen, color, p, r)


def draw_line(x0, y0, x1, y1, screen, screen_width, screen_height):
    p0 = pos_to_pix(x0, y0, screen_width, screen_height)
    p1 = pos_to_pix(x1, y1, screen_width, screen_height)
    pygame.draw.line(screen, BLACK, p0, p1)


def draw_lines(xs, ys, screen, screen_width, screen_height):
    for i in range(len(xs) - 1):
        draw_line(xs[i], ys[i], xs[i + 1], ys[i + 1], screen, screen_width, screen_height)

def rms(prev_pos: [MeasuredPos], filtered_pos):
    if len(prev_pos) == 0:
        return

    prev_err = 0
    filtered_err = 0
    for i in range(len(prev_pos)):
        prev_err += np.linalg.norm(prev_pos[i].noisy_pos_get() - prev_pos[i].true_pos_get())**2
        filtered_err += np.linalg.norm(filtered_pos[i].noisy_pos_get() - prev_pos[i].true_pos_get())**2
    
    prev_err /= len(prev_pos)
    filtered_err /= len(prev_pos)

    return prev_err**0.5, filtered_err**0.5



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
