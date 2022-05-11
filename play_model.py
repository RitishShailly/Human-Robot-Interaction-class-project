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
        print(type(state))
        print(state)
        s_tensor = torch.FloatTensor(state)
        action = self.model.encoder(s_tensor).detach().numpy()
        return action

def main():

    """Minimal code to teleop panda in pybullet env using joystick"""
    parser = argparse.ArgumentParser(description='Collecting online demonstrations')
    parser.add_argument('--trial', type=int, default=0)
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
    

    run = False
    shutdown = False
    record = False
    data = []
    # n_samples = 100
    last_time = time.time()
    xyz_array = np.empty((3, 0), float)
    while True:
        state = env.state()
        s = state['ee_position']
        input_state_list = s.tolist() + corners
        input_state = np.asarray(input_state_list)
        print(input_state) 
        a = model.policy(input_state) * 100.0
        print(a)

        if np.linalg.norm(a) > ACTION_SCALE:
            a = a / np.linalg.norm(a) * ACTION_SCALE
            
        u, A, B, start, X, Y = joystick.input()
        
        if A:
            run = True
        if B:
            run = False
            shutdown = True
            time_stop = time.time()
        if not run:
            a = np.asarray([0.0] * 3)
        # if not run and shutdown and time.time() - time_stop > 2.0:
            # print("x positions are: \n", xy_array[0])
            # print("y positions are: \n", xy_array[1])
            # plt.plot(xy_array[0], xy_array[1])
            # plt.savefig('results/play/corners_included_17t.pdf')
            # plt.xlabel('x-axis')
            # plt.ylabel('y-axis')
            # np.savetxt('results/play/corners_included_17t.csv', xy_array, delimiter=",")
            

        
        env.step(0.1 * a)
   

if __name__ == "__main__":
    main()

