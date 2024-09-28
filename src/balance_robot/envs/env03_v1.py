import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from balance_robot.envs.RobotBaseEnv import RobotBaseEnv, WHEEL_SPEED_DELTA_MAX

"""
Fires a block at the robot continuously from all directions

Fun to watch, but not great for training. There's too much going on for the
robot to learn anything. Due to geometry, a lot of the hits are easy to take
such as when it hits the sides. Problematic hits are the repeated shots hitting
the front/back and same side repeatedly. This is what's done in Env03_v2.
"""
class Env03(RobotBaseEnv):

    def __init__(self, **kwargs):
        RobotBaseEnv.__init__(self, 'env03_v1.xml', **kwargs)

        self.block_delay_time_start = None
        # time between each block firing
        self.block_delay = 0.0

    def step(self, a):
        reward = self._get_reward()

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + a[0] * WHEEL_SPEED_DELTA_MAX
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + a[1] * WHEEL_SPEED_DELTA_MAX

        self.data.actuator('motor_l_wheel').ctrl = [vel_l]
        self.data.actuator('motor_r_wheel').ctrl = [vel_r]
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)

        self._update_camera_follow()

        block_vel_vec = np.array(self.data.joint('block_joint').qvel[0:3])
        if (np.linalg.norm(block_vel_vec) < 0.1 and self.block_delay_time_start is None):
            self.remove_block()
            self.block_delay_time_start = self.data.time

        if (
            (self.block_delay_time_start is not None) and 
            (self.data.time - self.block_delay_time_start) > self.block_delay
            ):
            self.set_block_pos_vel()
            self.block_delay_time_start = None

        # terminate if pitch is greater than 50deg
        terminated = np.abs(self.get_pitch()) > (50 * math.pi / 180)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qpos[2] = 0

        # face a random direction
        x_rot = (np.random.random() - 0.5) * 2 * math.pi
        # rotate and pitch slightly
        y_rot = (np.random.random() - 0.5) * 0.4
        z_rot = (np.random.random() - 0.5) * 0.4
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        qpos[3:7] = rotation.as_quat()

        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        self.set_block_pos_vel()
        self.block_delay_time_start = None

        return self._get_obs()

    def remove_block(self):
        self.data.joint('block_joint').qpos[0:3] = np.array([10, 10, 0])

    def set_block_pos_vel(self):
        robot_pos = self.data.body("robot_body").xpos

        block_attack_angle = np.random.random() * 2 * math.pi
        block_x_pos = 0.3 * math.sin(block_attack_angle) + robot_pos[0]
        block_y_pos = 0.3 * math.cos(block_attack_angle) + robot_pos[1]

        block_pos = np.array([block_x_pos, block_y_pos, 0.15])
        block_target_pos = np.array([
            (np.random.random() - 0.5) * 0.06 + robot_pos[0],
            0 + robot_pos[1],
            np.random.random() * 0.075 + 0.1
        ])

        block_vel_vector =  block_target_pos - block_pos
        block_vel_vector = 5 * (block_vel_vector / np.linalg.norm(block_vel_vector))

        # face a random direction
        x_rot = np.random.random() * 2 * math.pi
        y_rot = np.random.random() * 2 * math.pi
        z_rot = np.random.random() * 2 * math.pi
        euler_angles = [x_rot, y_rot, z_rot]
        block_rotation = Rotation.from_euler('xyz', euler_angles)

        self.data.joint('block_joint').qpos[0:3] = block_pos
        self.data.joint('block_joint').qpos[3:7] = block_rotation.as_quat()
        self.data.joint('block_joint').qvel[0:3] = block_vel_vector
