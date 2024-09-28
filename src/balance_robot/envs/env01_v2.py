import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from balance_robot.envs.RobotBaseEnv import RobotBaseEnv, WHEEL_SPEED_DELTA_MAX
from balance_robot.envs.env01_v1 import Env01

"""
Same basic 'just learn to balance' env as env01_v1, but with noise added
to pitch + pitch_dot, and bigger variations in initial lean angle
"""
class Env01_v2(Env01):

    def get_pitch(self) -> float:
        p = super().get_pitch()
        # add some noise to pitch
        p += (np.random.random() - 0.5) * 0.05
        return p

    def get_pitch_dot(self) -> float:
        p = super().get_pitch_dot()
        # add some noise to pitch
        p += (np.random.random() - 0.5) * 0.05
        return p

    def step(self, a):
        reward = self._get_reward()

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + a[0] * WHEEL_SPEED_DELTA_MAX
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + a[1] * WHEEL_SPEED_DELTA_MAX

        # print(self.data.joint('torso_l_wheel').qvel[0])
        self.data.actuator('motor_l_wheel').ctrl = [vel_l]
        self.data.actuator('motor_r_wheel').ctrl = [vel_r]
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        # print(self.data.joint('torso_l_wheel').qvel[0])

        self._update_camera_follow()

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
        y_rot = (np.random.random() - 0.5) * 0.2
        z_rot = (np.random.random() - 0.5) * 2.0
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        qpos[3:7] = rotation.as_quat()

        qvel = self.init_qvel

        self.set_state(qpos, qvel)
        return self._get_obs()

