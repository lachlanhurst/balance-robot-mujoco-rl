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


class Env02(MujocoEnv, utils.EzPickle):
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
        observation_space = Box(
            np.array([-math.pi, -math.pi * 2,  -80, -80]), 
            np.array([math.pi, math.pi * 2, 80, 80]),
            dtype=np.float32
        )
        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent.joinpath('env02_v1.xml')),
            20,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _set_action_space(self):
        # called by init in parent class
        v_mag = 2.0
        self.action_space = Box(
            np.array([-v_mag, -v_mag]), 
            np.array([v_mag, v_mag]),
            dtype=np.float32
        )
        return self.action_space

    def step(self, a):
        reward = 1.0

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + a[0]
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + a[1]

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
        ob = self._get_obs()
        terminated = np.abs(ob[0]) > (50 * math.pi / 180)
        if self.render_mode == "human":
            self.render()
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

        floor = self.model.geom('floor')
        l_wheel = self.model.geom('l_wheel_geom')
        r_wheel = self.model.geom('r_wheel_geom')

        # to get a random friction value between 0.5 and 1.0
        friction = np.random.random() / 2 + 0.5
        floor.friction[0] = friction
        l_wheel.friction[0] = friction
        r_wheel.friction[0] = friction

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

        return np.array(
            # [pitch, pitch_dot, yaw_dot, wheel_vel_l, wheel_vel_r],
            [pitch, pitch_dot, wheel_vel_l, wheel_vel_r],
            dtype=np.float32
        ).ravel()

