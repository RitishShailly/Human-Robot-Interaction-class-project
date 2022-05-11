import sys
import time
import numpy as np
import pygame
from utils import JoystickControl
import copy
from env import PandaEnv
import pickle
import argparse


#HOME = [-1.57, -1.506, -2.22, -1.01, 1.56, 3.14]

# HOME = [4.73, -1.59, -2.12, -0.97, 1.54, -0.02]
# corner1 = [-1.551, -2.193, -1.388, -1.041, 1.571, -0.0196]
# corner2 = [-1.198, -2.194, -1.361, -1.041, 1.571, -0.01964]

def main():
    """Minimal code to teleop panda in pybullet env using joystick"""
    parser = argparse.ArgumentParser(description='Collecting offline demonstrations')
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()
    # Create simple panda environment
    env = PandaEnv()
    # Create joystick
    joystick = JoystickControl()
    # steptime(ms)
    steptime = 0.05
    # scaling factor for translation and rotation
    scaling_trans = 0.3
    scaling_rot = 0.2
    # reset the robot to home state, input is a 7x1 list of joint angles
    env.reset()
    # saving data
    data = []
    record = False
    last_time = time.time()
    # file for storing demo
    filename = "demos/move/trial" + str(args.trial) + ".pkl"
    while True:
        # Get state of robot at every time step
        # state is a dictionary, access using keys
        state = env.state()
        print(state['joint_position'])
        # u = joystick input (x, y, z)
        # start = A_pressed
        # mode = B_pressed
        # stop = START_pressed
        # X_in = X_pressed
        # Y_in = Y_pressed
        u, A_in, B_in, stop, X_in, Y_in = joystick.input()
        # if start pressed, quit
        if stop:
            pickle.dump(data, open(filename, "wb"))
            print(data)
            break
        # Start recording when A pressed
        if A_in and record == False:
            record = True
            print("[*] Recording Started...")
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
        # record states
        curr_time = time.time()
        s = state["joint_position"][:7]
        if record and curr_time - last_time > steptime:
            data.append(s.tolist())
            last_time = time.time()

        # Send velocities to robot
        env.step(0.1 * xdot[:3])

if __name__ == "__main__":
    main()


