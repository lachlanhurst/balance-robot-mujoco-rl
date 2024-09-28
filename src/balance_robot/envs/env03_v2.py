import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from balance_robot.envs.RobotBaseEnv import RobotBaseEnv, WHEEL_SPEED_DELTA_MAX
from balance_robot.envs.env03_v1 import Env03

"""
Block fired continuously targeted at the front (or back) only. This is a better
training env than Env03_v1 as it is consistently the worst case.
"""
class Env03_v2(Env03):

    def __init__(self, **kwargs):
        Env03.__init__(self, **kwargs)

        # Choose a side and continuously fire blocks
        # that side. If alternating sides during a session,
        # the block hits can slow the robot, helping it.
        self.attack_side_front = np.random.random() > 0.5
        self.block_delay = 0.5

    def set_block_pos_vel(self):
        robot_pos = self.data.body("robot_body").xpos

        block_attack_angle = -self.get_yaw()
        if not self.attack_side_front:
            block_attack_angle += math.pi

        block_x_pos = 0.3 * math.sin(block_attack_angle) + robot_pos[0]
        block_y_pos = 0.3 * math.cos(block_attack_angle) + robot_pos[1]

        block_pos = np.array([block_x_pos, block_y_pos, 0.15])
        # block_target_pos = np.array([
        #     (np.random.random() - 0.5) * 0.06 + robot_pos[0],
        #     0 + robot_pos[1],
        #     np.random.random() * 0.075 + 0.1
        # ])
        block_target_pos = np.array([
            (np.random.random() - 0.5) * 0.02 + robot_pos[0],
            0 + robot_pos[1],
            np.random.random() * 0.025 + 0.13
        ])

        block_vel_vector =  block_target_pos - block_pos
        block_vel_vector = 7.5 * (block_vel_vector / np.linalg.norm(block_vel_vector))

        # face a random direction
        x_rot = np.random.random() * 2 * math.pi
        y_rot = np.random.random() * 2 * math.pi
        z_rot = np.random.random() * 2 * math.pi
        euler_angles = [x_rot, y_rot, z_rot]
        block_rotation = Rotation.from_euler('xyz', euler_angles)

        self.data.joint('block_joint').qpos[0:3] = block_pos
        self.data.joint('block_joint').qpos[3:7] = block_rotation.as_quat()
        self.data.joint('block_joint').qvel[0:3] = block_vel_vector
