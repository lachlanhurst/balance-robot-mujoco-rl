import math
import numpy as np
import pathlib

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class Env01(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        observation_space = Box(
            np.array([-math.pi, -math.pi, -80, -80]), 
            np.array([math.pi, math.pi, 80, 80]),
            dtype=np.float32
        )
        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent.joinpath('scene.xml')),
            20,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _set_action_space(self):
        # called by init in parent class
        v_mag = 4.0
        self.action_space = Box(
            np.array([-v_mag, -v_mag]), 
            np.array([v_mag, v_mag]),
            dtype=np.float32
        )
        return self.action_space

    def step(self, a):
        reward = 1.0

        # reward -= (self.data.body("l_wheel").xpos[2] - 0.034)
        # reward -= (self.data.body("r_wheel").xpos[2] - 0.034)

        bounced = self.data.body("l_wheel").xpos[2] > 0.045 or self.data.body("r_wheel").xpos[2] > 0.045

        # print("a")
        # print(a)

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + a[0]
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + a[1]

        a[0] = vel_l
        a[1] = vel_r

        # print(self.data.joint('torso_l_wheel').qvel[0])
        self.do_simulation(a, self.frame_skip)
        # print(self.data.joint('torso_l_wheel').qvel[0])
        ob = self._get_obs()
        terminated = np.abs(ob[0]) > 0.5 or bounced
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

    def _get_obs(self):
        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot()
        wheel_vel_l, wheel_vel_r = self.get_wheel_velocities()

        return np.array([pitch, pitch_dot, wheel_vel_l, wheel_vel_r]).ravel()
        # return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
