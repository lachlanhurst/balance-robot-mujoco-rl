import math
import numpy as np
import pathlib

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 2.04,
    "elevation": -25,
    "azimuth": 45,
}

# PITCH_MAX = math.pi
# PITCH_DOT_MAX = math.pi * 2
# WHEEL_SPEED_MAX = 80.0
# WHEEL_SPEED_DELTA_MAX = 2.0
PITCH_MAX = 1
PITCH_DOT_MAX = 1
WHEEL_SPEED_MAX = 80.0
WHEEL_SPEED_DELTA_MAX = 2.0


class Env01(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        # normalize the observation space (well the wheel speeds for now) to
        # better support quantization later
        observation_space = Box(
            # np.array([-1.0, -1.0, -1.0, -1.0]),
            # np.array([ 1.0,  1.0,  1.0,  1.0]),
            np.array([-math.pi, -math.pi * 2,  -1.0, -1.0]),
            np.array([math.pi, math.pi * 2, 1.0, 1.0]),
            dtype=np.float32
        )

        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent.joinpath('env01_v1.xml')),
            20,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=800,
            height=800,
            **kwargs,
        )

    def _set_action_space(self):
        # called by init in parent class
        # normalize the action space to better support quantization later
        self.action_space = Box(
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            dtype=np.float32
        )
        return self.action_space

    def step(self, a):
        reward = 1.0

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + a[0] * WHEEL_SPEED_DELTA_MAX
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + a[1] * WHEEL_SPEED_DELTA_MAX

        a[0] = vel_l
        a[1] = vel_r

        target_velocity = 0
        target_yaw_dot = 0

        average_wheel_speed = (vel_l * -1 + vel_r) / 2.0
        dv = target_velocity - average_wheel_speed
        # reward -= 0.05 * abs(dv)

        dyd = target_yaw_dot - self.get_yaw_dot()
        reward -= 0.025 * abs(dyd)

        # print(self.data.joint('torso_l_wheel').qvel[0])
        self.do_simulation(a, self.frame_skip)
        # print(self.data.joint('torso_l_wheel').qvel[0])

        self._update_camera_follow()

        # terminate if pitch is greater than 50deg
        terminated = np.abs(self.get_pitch()) > (50 * math.pi / 180)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}

    def _update_camera_follow(self):
        # get robot position
        pos = self.data.body("robot_body").xpos
        if self.unwrapped.mujoco_renderer.viewer is not None:
            # viewer is None first time render is called!
            v = self.unwrapped.mujoco_renderer.viewer
            # Adjust the camera to follow the robot
            v.cam.lookat[:] = pos
            v.cam.distance = 1.0

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

    def get_wheel_velocities(self) -> float:
        vel_m_0 = self.data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = self.data.joint('torso_r_wheel').qvel[0]

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        return (vel_m_0, vel_m_1)

    def get_yaw_dot(self):
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        return angular[2]

    def _get_obs(self):
        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot()
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
