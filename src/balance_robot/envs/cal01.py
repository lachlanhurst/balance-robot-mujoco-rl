import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from balance_robot.envs.RobotBaseEnv import RobotBaseEnv, WHEEL_SPEED_DELTA_MAX


class Cal01(RobotBaseEnv):

    def __init__(self, **kwargs):
        RobotBaseEnv.__init__(self, 'env01_v1.xml', **kwargs)

    def step(self, a):
        reward = self._get_reward()

        # print(self.data.joint('torso_l_wheel').qvel[0])
        self.data.actuator('motor_l_wheel').ctrl = [20]
        self.data.actuator('motor_r_wheel').ctrl = [20]
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        # print(self.data.joint('torso_l_wheel').qvel[0])

        self._update_camera_follow()

        vel_l = self.data.joint('torso_l_wheel').qvel[0]
        vel_r = self.data.joint('torso_r_wheel').qvel[0]

        # terminate if pitch is greater than 50deg
        print(f"{self.data.time}, {vel_l}, {vel_r}")
        terminated = self.data.time > 1.0
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos
        qpos[2] = 0.15

        x_rot = 0
        y_rot = 0 
        z_rot = math.pi
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        qpos[3:7] = rotation.as_quat()

        qvel = self.init_qvel

        self.set_state(qpos, qvel)
        return self._get_obs()

