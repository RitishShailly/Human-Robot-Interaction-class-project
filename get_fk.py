import numpy as np
import rospy
import sys
import time
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import pickle
import os



class Robot(object):

    def __init__(self):
        # Action client for joint move commands
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)


    def forward_kinematics(self, s, end_link="wrist_3_link", base_link="base_link"):

    	return self.kdl_kin.forward(s)




folder = "demos/"
ur10 = Robot()

for filename in os.listdir(folder):
	traj = pickle.load(open(folder + "/" + filename, "rb"))
	traj1 = []
	for state in traj:
		pose = ur10.forward_kinematics(state)
		traj1.append([np.asarray(state), np.asarray(pose)])

	pickle.dump(traj1, open(folder + "/" + filename + "_fk.pkl", "wb"))
