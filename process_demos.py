import pickle
import numpy as np
import os, sys
from env import PandaEnv, BALL_POSE



folder = "demos/trial"

noise = 0.01        
n_upsamples = 10
n_lookahead = 1

# BALL_POSE = [[0.4, -0.8, 0], [0.8, -0.8, 0], [0.8, -0.3, 0], [0.4,-0.3, 0]]

# joint positions at corner points
offset = [0, 0, 0.1]
corner1 = [x+y for x,y in zip(BALL_POSE[0],offset)]
corner2 = [x+y for x,y in zip(BALL_POSE[1],offset)]
corner3 = [x+y for x,y in zip(BALL_POSE[2],offset)]
corner4 = [x+y for x,y in zip(BALL_POSE[3],offset)]


sapairs = []
for filename in os.listdir(folder):
    traj = pickle.load(open(folder + "/" + filename, 'rb'))
    print("I am loading file: ", folder + "/" + filename)
    print("it has this many data points: ", len(traj))
    traj = np.asarray(traj)
    for idx in range(len(traj) - n_lookahead):
        s_base = traj[idx]
        sp = traj[idx + n_lookahead]
        for _ in range(n_upsamples):
            s = np.copy(s_base) + np.random.normal(0, noise, len(s_base))
            a = sp - s
            # print(s)
            # print(type(a), type(s))
            sapairs.append(s.tolist() + corner1+ corner2+ corner3+ corner4 + a.tolist())
            # sapairs.append(s.tolist() + a.tolist())

pickle.dump(sapairs, open("data/sa_pairs.pkl", "wb"))
print("I have this many state-action pairs: ", len(sapairs))
# print('Take a sneak peek at sapairs: ')
# print(sapairs[:10])
