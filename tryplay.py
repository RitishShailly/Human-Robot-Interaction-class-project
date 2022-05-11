import sys
import time
import numpy as np
import pygame
from utils import JoystickControl
import copy
from env import PandaEnv, BALL_POSE
import pickle
import torch
from collections import deque
import argparse
from train_model import BC
# import matplotlib.pyplot as plt

ACTION_SCALE = 0.15
MOVING_AVERAGE = 10


offset = [0, 0, 0.1]
corner1 = [x+y for x,y in zip(BALL_POSE[0],offset)]
corner2 = [x+y for x,y in zip(BALL_POSE[1],offset)]
corner3 = [x+y for x,y in zip(BALL_POSE[2],offset)]
corner4 = [x+y for x,y in zip(BALL_POSE[3],offset)]
corners = corner1 + corner2 + corner3 + corner4

class Model(object):
    def __init__(self):
        self.model = BC(32)
        model_dict = torch.load("models/MLP_model", map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def policy(self, state):
        s_tensor = torch.FloatTensor(state)
        action = self.model.encoder(s_tensor).detach().numpy()
        return action

def main():

    """Minimal code to teleop panda in pybullet env using joystick"""
    parser = argparse.ArgumentParser(description='Collecting online demonstrations')
    parser.add_argument('--correction', type=int, default=0)
    parser.add_argument('--traj', type=str, default='0')
    args = parser.parse_args()

    # Create simple panda environment
    env = PandaEnv()
    # Define State
    
    # Create joystick
    joystick = JoystickControl()
    # set the model
    model = Model()
    # steptime(ms)
    steptime = 0.05
    # scaling factor for translation and rotation
    scaling_trans = 0.3
    scaling_rot = 0.2
    # reset the robot to home state, input is a 7x1 list of joint angles
    env.reset()
    print("[*] Initialized, Moving Home")

    data = []
    run = False
    shutdown = False
    record = False
    # n_samples = 100
    last_time = time.time()
    xyz_array = np.empty((3, 0), float)
    xdot_h = np.zeros(6)

    trajectory = np.array([[0.0, 0.0, 0.0]])


    while True:
        state = env.state()
        s = state['ee_position']
        input_state_list = s.tolist() + corners
        input_state = np.asarray(input_state_list)
        # print(input_state) 
        a = model.policy(input_state) * 100.0
        # print(a)
        if np.linalg.norm(a) > ACTION_SCALE:
            a = a / np.linalg.norm(a) * ACTION_SCALE
        u, A, B, start, X, Y = joystick.input()

        if u != [0,0,0]:
            print(state['joint_position'])
            # check and toggle the state of A button
            A_state = 0
            if run == True:
                run = False
                A_state = 1

            xdot_h[:3] = scaling_trans * np.asarray(u)
            xdot = xdot_h
            # Stop from hitting the table
            if s[2] < 0.1 and xdot[2] < 0:
                xdot[2] = 0  
            # record states
            curr_time = time.time()
            if curr_time - last_time > steptime:
                data.append(s.tolist())
                last_time = time.time()
            env.step(0.1 * xdot[:3])
            if A_state == 1:
                run = True
                A_state = 0

        if A:
            run = True
        if B:
            run = False
            shutdown = True
            time_stop = time.time()
        if not run:
            a = np.asarray([0.0] * 3)
        if run:
            current_position = []
            current_position_1= state['ee_position'][0]
            current_position_2= state['ee_position'][1]
            current_position_3= state['ee_position'][2]
            current_position = [[current_position_1, current_position_2, current_position_3]]
            curr_pos = np.asarray(current_position)
            # current_position = np.array([current_position_1,current_position_2,current_position_3])
            # print('trajectory = ' , trajectory)
            # print('curr_pos = ', curr_pos)
            trajectory = np.concatenate((trajectory, curr_pos), axis = 0)
        if Y:
            file_str = "correction"
            filename = "demos/corrections/" + file_str + str(args.correction) + ".pkl"
            pickle.dump(data, open(filename, "wb"))
            np.savetxt('trajectory/traj'+ args.traj +'.csv', trajectory, delimiter=",")
            print("The corrections are stored in file:" + file_str + str(args.correction)+ ".pkl\n\n\n\n\n")
        if start:
            np.savetxt('trajectory/traj_'+ args.traj +'.csv', trajectory, delimiter=",")
            print('[*] SAVING TRAJECTORY FILE.....')
            break
        env.step(0.1 * a)
   

if __name__ == "__main__":
    main()

