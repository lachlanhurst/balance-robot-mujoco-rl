import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from balance_robot.envs.RobotBaseEnv import RobotBaseEnv, WHEEL_SPEED_DELTA_MAX
from balance_robot.envs.env01_v1 import Env01

"""
Env for training on moving backward and forward
"""
class Env01_v3(Env01):

    def __init__(self, **kwargs):
        Env01.__init__(self, **kwargs)

        self.delay_target_speed = 0.0
        self.delay_target_yaw = 0.0

    def step(self, a):
        if self.data.time > 5.5:
            self.target_wheel_speed = 3.0 * self.delay_target_speed
        elif self.data.time > 4.5:
            self.target_wheel_speed = 2.0 * self.delay_target_speed
        elif self.data.time > 3.0:
            self.target_wheel_speed = -1.0 * self.delay_target_speed
        elif self.data.time > 0.5:
            self.target_wheel_speed = self.delay_target_speed

        return Env01.step(self, a)

    def reset_model(self):
        self.target_wheel_speed = 0
        self.target_yaw = 0

        # between -10 and 10
        self.delay_target_speed = self.np_random.uniform(low=-10.0, high=10)
        # now between 10 to 20 or -20 to -10
        if self.delay_target_speed > 0:
            self.delay_target_speed += 10
        else:
            self.delay_target_speed -= 10

        return Env01.reset_model(self)

    def _get_reward(self):
        reward = 0.6

        pitch = self.get_pitch()
        wheel_speed = self.get_wheel_speed()
        dv = self.target_wheel_speed - wheel_speed

        # penalty for not being vertical 
        reward -= (abs(pitch) * 0.05)


        MAX_DV = 40.0
        max_dv = np.clip(dv, -MAX_DV, MAX_DV)
        # will be -1 to 1
        dv_n = max_dv / MAX_DV
        dv_s = abs(dv_n)

        reward -= (0.15 * dv_s)

        if self.target_wheel_speed > 0 and self.target_wheel_speed > wheel_speed:
            # then reward for leaning forward, needs to speed up forward
            reward += (-1.0*pitch) * 10.0 * dv_s
        elif self.target_wheel_speed < 0 and self.target_wheel_speed < wheel_speed:
            # then reward for leaning backwards, needs to speed up backwards
            reward += (1.0*pitch) * 10.0 * dv_s
        elif self.target_wheel_speed > 0 and self.target_wheel_speed < wheel_speed:
            # then reward for leaning backward, needs to slow down going forward
            reward += (1.0*pitch) * 10.0 * dv_s
        elif self.target_wheel_speed < 0 and self.target_wheel_speed > wheel_speed:
            # then reward for leaning backwards, needs to speed up backwards
            reward += (-1.0*pitch) * 10.0 * dv_s

        # if self.loop_count % 10 == 0:
        #     print(f"{dv_s}   {dv}  {reward}")

        # small penalty for uneven wheel speeds (turning)
        dyd = self.target_yaw - self.get_wheel_yaw()
        # print(0.04 * abs(dyd))
        reward -= (0.007 * abs(dyd))

        return reward
