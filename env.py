import os
import numpy as np
import pybullet as p
import pybullet_data
from panda import Panda
import pickle

BALL_POSE = [[0.4, -0.8, 0], [0.8, -0.8, 0], [0.8, -0.3, 0], [0.4,-0.3, 0]]

class PandaEnv():

    def __init__(self, sim_type="GUI"):
        # create simulation (GUI)
        
        self.urdfRootPath = pybullet_data.getDataPath()

        # GUI launches pybullet with visualization, DIRECT creates env with no visualization
        if sim_type == "GUI":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)

        # set up camera
        self._set_camera()

        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, -0.6, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "sphere_small.urdf"), basePosition= BALL_POSE[0], useFixedBase = 1)
        p.loadURDF(os.path.join(self.urdfRootPath, "sphere_small.urdf"), basePosition= BALL_POSE[1], useFixedBase = 1)
        p.loadURDF(os.path.join(self.urdfRootPath, "sphere_small.urdf"), basePosition= BALL_POSE[2], useFixedBase = 1)
        p.loadURDF(os.path.join(self.urdfRootPath, "sphere_small.urdf"), basePosition= BALL_POSE[3], useFixedBase = 1)
        print(self.urdfRootPath)
        # load a panda robot
        self.panda = Panda([0, -0.6, 0])

    # def reset(self, q=[0.0, -np.pi/4, 0.0, -2*np.pi/4, 0.0, np.pi/4, np.pi/4]):
    def reset(self, q=[ 1.24320905, -1.83260367, -1.20135715, -2.47015842, -1.44708791,  1.13748933,  1.33526437]):
    
        """Reset panda to a given position"""
        self.panda.reset(q)
        return [self.panda.state]

    def close(self):
        """Disconnects the pybullet environment and closes simulation"""
        p.disconnect()

    def step(self, action):
        """ Takes given action in the environment. Actions are in cartesian space action = [x, y, z, r, p, y]"""
        # get current state
        state = [self.panda.state]
        self.panda.step(dposition=action[:3])

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        next_state = [self.panda.state]
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    def state(self):
        return self.panda.state

    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                     height=self.camera_height,
                                                                     viewMatrix=self.view_matrix,
                                                                     projectionMatrix=self.proj_matrix)
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        # p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=90, cameraPitch=-31.4,
        #                              cameraTargetPosition=[1.1, 0.0, 0.0])
        # self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
        #                                                        distance=1.0,
        #                                                        yaw=90,
        #                                                        pitch=-50,
        #                                                        roll=0,
        #                                                        upAxisIndex=2)
        p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw = -0.8, cameraPitch=-89.8,
                                     cameraTargetPosition=[0.41, -0.47, -0.51])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)
