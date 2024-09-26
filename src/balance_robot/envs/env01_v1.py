import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from balance_robot.envs.RobotBaseEnv import RobotBaseEnv, WHEEL_SPEED_DELTA_MAX


class Env01(RobotBaseEnv):

    def __init__(self, **kwargs):
        RobotBaseEnv.__init__(self, 'env01_v1.xml', **kwargs)

    def step(self, a):
        reward = 1.0

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + a[0] * WHEEL_SPEED_DELTA_MAX
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + a[1] * WHEEL_SPEED_DELTA_MAX

        target_velocity = 0
        target_yaw_dot = 0

        average_wheel_speed = (vel_l * -1 + vel_r) / 2.0
        dv = target_velocity - average_wheel_speed
        # reward -= 0.05 * abs(dv)

        dyd = target_yaw_dot - self.get_yaw_dot()
        reward -= 0.025 * abs(dyd)

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
        y_rot = (np.random.random() - 0.5) * 0.4
        z_rot = (np.random.random() - 0.5) * 0.4
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        qpos[3:7] = rotation.as_quat()

        qvel = self.init_qvel

        self.set_state(qpos, qvel)
        return self._get_obs()

