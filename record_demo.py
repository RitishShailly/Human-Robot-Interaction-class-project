import sys
import time
import numpy as np
import pygame
from utils import JoystickControl
import copy
from env import PandaEnv, BALL_POSE
import pickle
import argparse


corner1 = BALL_POSE[0]
corner2 = BALL_POSE[1]
corner3 = BALL_POSE[2]
corner4 = BALL_POSE[3]


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
    scaling_trans = 0.15
    scaling_rot = 0.2
    # reset the robot to home state, input is a 7x1 list of joint angles
    env.reset()
    # saving data
    data = []
    Y_pressed = 0
    record = False
    last_time = time.time()
    # file for storing demo
    filename = "demos/move/trial" + str(args.trial) + ".pkl"
    

    
    while True:
        # Get state of robot at every time step
        # state is a dictionary, access using keys
        state = env.state()
        # print(state['ee_position'])
        # print(state['joint_position'])
        # u = joystick input (x, y, z)
        # start = A_pressed
        # mode = B_pressed
        # stop = START_pressed
        # X_in = X_pressed
        # Y_in = Y_pressed
        u, A_in, B_in, stop, X_in, Y_in = joystick.input()
        # if start pressed, quit
        if stop:
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
        xyz = state['ee_position']
        # Stop from hitting the table
        if xyz[2] < 0.1 and xdot[2] < 0:
            xdot[2] = 0  
        # record states
        curr_time = time.time()
        # print(state['ee_position'])
        s = state["joint_position"][:7]
        if Y_in:
            print(state['joint_position'][:7])
        if B_in:
            pickle.dump(data, open(filename, "wb"))
            print('[*] Saving Recorded Data... \n\n\n\n')
            trajectory = np.asarray(data)
            # print(trajectory)
            np.savetxt('record_traj/recording'+ str(args.trial) +'.csv', trajectory, delimiter=",")
            record = False
            print('[*]STOPPED RECORDING....')
        if record and curr_time - last_time > steptime:
            # data.append(s.tolist())
            print(state['ee_position'][:3])
            data.append(xyz.tolist())
            last_time = time.time()

        # Send velocities to robot
        env.step(0.1 * xdot[:3])

if __name__ == "__main__":
    main()


