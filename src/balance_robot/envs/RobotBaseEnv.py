import math
import mujoco
import numpy as np
import pathlib

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 1.25,
    "elevation": -25,
    "azimuth": 45,
}

PITCH_MAX = 0.25
PITCH_DOT_MAX = 1
WHEEL_SPEED_MAX = 170.0
WHEEL_SPEED_DELTA_MAX = 4.0

"""
Most of the Env code will be common across different scenarios as the robot
doesn't change. The base class includes all this common code.
"""
class RobotBaseEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    def __init__(self, env_filename: str, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        observation_space = Box(
            np.array([-math.pi * 2, -math.pi * 2,  -1.0, -1.0]),
            np.array([math.pi * 2, math.pi * 2, 1.0, 1.0]),
            dtype=np.float32
        )

        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent.joinpath(env_filename)),
            250,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=800,
            height=800,
            **kwargs,
        )

        self.last_time = None
        self.last_pitch = None

    def _set_action_space(self):
        # called by init in parent class
        # normalize the action space to better support quantization later
        self.action_space = Box(
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            dtype=np.float32
        )
        return self.action_space

    def _update_camera_follow(self):
        # get robot position
        pos = self.data.body("robot_body").xpos
        if self.unwrapped.mujoco_renderer.viewer is not None:
            # viewer is None first time render is called!
            v = self.unwrapped.mujoco_renderer.viewer
            # Adjust the camera to follow the robot
            v.cam.lookat[:] = pos
            v.cam.distance = 1.25

    def render(self):
        if self.mujoco_renderer.viewer is not None and self.render_mode == 'human':
            # in this case it's a Gymnasium Mujoco Viewer
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Pitch",
                text2="{:.2f}".format(self.get_pitch() * 180 / math.pi)
            )
        return super().render()

    def get_pitch(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Quaternion order is [x, y, z, w]
        angles = rotation.as_euler('xyz', degrees=False)
        # print(angles)
        return angles[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        # print(angular)
        return angular[0]

    def get_pitch_dot_alt(self) -> float:
        # alternate method of calculating pitch dot, this is how the real
        # robot does it
        pitch = self.get_pitch()
        ts = self.data.time

        pitch_dot = 0
        if self.last_time is not None and self.last_pitch is not None:
            dt = ts - self.last_time
            pitch_dot = (pitch - self.last_pitch) / dt
        
        self.last_time = ts
        self.last_pitch = pitch

        return pitch_dot

    def get_wheel_velocities(self) -> float:
        vel_m_0 = self.data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = self.data.joint('torso_r_wheel').qvel[0]

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        return (vel_m_0, vel_m_1)

    def get_yaw(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Quaternion order is [x, y, z, w]
        angles = rotation.as_euler('xyz', degrees=False)
        return angles[2]

    def get_yaw_dot(self):
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        return angular[2]

    def _get_reward(self):
        reward = 1.0

        target_velocity = 0
        target_yaw_dot = 0

        vel_l = self.data.joint('torso_l_wheel').qvel[0]
        vel_r = self.data.joint('torso_r_wheel').qvel[0]

        average_wheel_speed = (vel_l * -1 + vel_r) / 2.0
        dv = target_velocity - average_wheel_speed
        # penalizing wheel velocity is problematic as sometimes
        # we need to increase speed to maintain balance.
        # reward -= 0.05 * abs(dv)

        # small penalty for uneven wheel speeds (turning)
        dyd = target_yaw_dot - self.get_yaw_dot()
        reward -= (0.025 * abs(dyd))

        # penalty for not being vertical 
        pitch = self.get_pitch()
        reward -= (abs(pitch))

        # penalty for leaning forward when moving forward (and vice-versa)
        # will also give reward for leaning back when moving forward
        # this the main reward component that gets the robot to slow down 
        r = pitch * dv * 0.5
        reward += r

        return reward

    def _get_obs(self):
        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot_alt()
        yaw_dot = self.get_yaw_dot()
        wheel_vel_l, wheel_vel_r = self.get_wheel_velocities()

        pitch_normalized = pitch / PITCH_MAX
        pitch_dot_normalized = pitch_dot / PITCH_DOT_MAX
        wheel_vel_l_normalized = wheel_vel_l / WHEEL_SPEED_MAX
        wheel_vel_r_normalized = wheel_vel_r / WHEEL_SPEED_MAX

        return np.array(
            [
                pitch_normalized,
                pitch_dot_normalized,
                wheel_vel_l_normalized,
                wheel_vel_r_normalized
            ],
            dtype=np.float32
        ).ravel()

        # obs_clip = np.clip(obs, a_min = -1.0, a_max = 1.0)
        # print(obs_clip)
        # return obs_clip
