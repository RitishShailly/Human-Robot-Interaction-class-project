import numpy as np
import pygame
import time
import os

class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1
        self.timeband = 0.5
        self.lastpress = time.time()
        self.z_axis = 4

    def input(self):
        pygame.event.get()
        curr_time = time.time()
        dx = self.gamepad.get_axis(0)
        dy = -self.gamepad.get_axis(1)
        dz = -self.gamepad.get_axis(self.z_axis)
        if abs(dx) < self.deadband:
            dx = 0.0
        if abs(dy) < self.deadband:
            dy = 0.0
        if abs(dz) < self.deadband:
            dz = 0.0
        A_pressed = self.gamepad.get_button(0) #and (curr_time - self.lastpress > self.timeband)
        B_pressed = self.gamepad.get_button(1) #and (curr_time - self.lastpress > self.timeband)
        X_pressed = self.gamepad.get_button(2)
        Y_pressed = self.gamepad.get_button(3)
        START_pressed = self.gamepad.get_button(7) #and (curr_time - self.lastpress > self.timeband)
        if A_pressed or B_pressed or START_pressed:
            self.lastpress = curr_time
        return [dx, dy, dz], A_pressed, B_pressed, START_pressed, X_pressed, Y_pressed

    def quit(self):
        self.gamepad.quit()