import time
import numpy as np
import pygame
from env import PandaEnv
from utils import JoystickControl

def teleop():
    """Minimal code to teleop panda in pybullet env using joystick"""
    
    # Create simple panda environment
    env = PandaEnv()
    # Create joystick
    joystick = JoystickControl()
    # steptime(ms)
    steptime = 0.1
    # scaling factor for translation and rotation
    scaling_trans = 0.1
    scaling_rot = 0.2
    # reset the robot to home state, input is a 7x1 list of joint angles
    env.reset()

    startup = True
    while True:
        # Get state of robot at every time step
        # state is a dictionary, access using keys
        state = env.state()
        # u = joystick input (x, y, z)
        # start = A_pressed
        # mode = B_pressed
        # stop = START_pressed
        # X_in = X_pressed
        # Y_in = Y_pressed
        u, start, mode, stop, X_in, Y_in = joystick.input()
        # if start pressed, quit
        if mode:
            joystick.quit()
            break
        # Cartesian velocities by user=[x, y, z, r, p, y]
        xdot_h = np.zeros(6)
        # Use joystick input to update cartesian velocities
        xdot_h[:3] = scaling_trans * np.asarray(u)
        # Final velocities for the robot
        xdot = xdot_h
        # Get position of end effector
        x_pos = state['ee_position']
        # Stop from hitting the table
        if x_pos[2] < 0.1 and xdot[2] < 0:
            xdot[2] = 0  
        # Send velocities to robot
        env.step(steptime * xdot[:3])


def main():
    teleop()

if __name__ == '__main__':
    main()