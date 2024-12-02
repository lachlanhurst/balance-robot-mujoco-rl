import math
import mujoco
import numpy as np
import pathlib
import tensorflow as tf

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
YAW_MAX = 45.0

"""
Most of the Env code will be common across different scenarios as the robot
doesn't change. The base class includes all this common code.
"""
class RobotMoveBaseEnv(MujocoEnv, utils.EzPickle):
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

        # order of observation space
        # - normalised speed
        # - normalised yaw
        # - 8 distance observations from ray casting (lidar on real robot)
        observation_space = Box(
            np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([ 1.0,  1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
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

        self.loop_count = 0
        self.last_time = None
        self.last_pitch = None

        self.target_wheel_speed = 0.0
        self.target_yaw = 0.0

        # setup the lidar ray directions, robot location coordinate system
        middle_ray_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        rotation_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angles =  np.arange(-50, 50.1, 14.285) * (math.pi / 180)
        self.ray_directions = []
        for angle in angles:
            rotation = Rotation.from_rotvec(angle * rotation_axis)
            rotated_vector = rotation.apply(middle_ray_direction)
            self.ray_directions.append(rotated_vector)

        # load the tflite model and setup inputs/outputs
        p = pathlib.Path(__file__).parent
        p = p / 'RobotMovePolicy.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=str(p))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        # there's only one input tensor
        input_detail = self.input_details[0]
        # get the quantization input params
        self.input_scale, self.input_zero_point = input_detail['quantization']

        self.output_details = self.interpreter.get_output_details()
        # unique to the PPO policy, which has multiple outputs
        # the second output is the one that includes the actions needed
        output_detail = self.output_details[1]
        self.output_scale, self.output_zero_point = output_detail['quantization']

    def _set_action_space(self):
        # called by init in parent class
        # normalize the action space to better support quantization later

        # order of action space
        # difference in target wheel speed to actual wheel speed
        # difference in target yaw (yaw being difference in LHS vs RHS wheel speeds) to actual yaw
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
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Speed",
                text2="{:.2f}".format(self.get_wheel_speed())
            )
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Target",
                text2="{:.2f}".format(self.target_wheel_speed)
            )
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Yaw",
                text2="{:.2f}".format(self.get_wheel_yaw())
            )
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Target Yaw",
                text2="{:.2f}".format(self.target_yaw)
            )
        return super().render()


    def _get_move_obs(self):
        # returns the observations for the move/balance tflite model
        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot_alt()
        wheel_vel_l, wheel_vel_r = self.get_wheel_velocities()

        pitch_normalized = pitch / PITCH_MAX
        pitch_dot_normalized = pitch_dot / PITCH_DOT_MAX
        wheel_vel_l_normalized = wheel_vel_l / WHEEL_SPEED_MAX * 4
        wheel_vel_r_normalized = wheel_vel_r / WHEEL_SPEED_MAX * 4
        delta_wheel_speed_normalized = (self.target_wheel_speed - self.get_wheel_speed()) / WHEEL_SPEED_MAX * 4
        delta_wheel_yaw_normalized = (self.target_yaw - self.get_wheel_yaw()) / YAW_MAX * 3

        return np.array(
            [
                pitch_normalized,
                pitch_dot_normalized,
                wheel_vel_l_normalized,
                wheel_vel_r_normalized,
                delta_wheel_speed_normalized,
                delta_wheel_yaw_normalized
            ],
            dtype=np.float32
        ).ravel()

    def _step_wheel_speeds(self, target_speed: float, target_yaw: float) -> None:
        self.target_wheel_speed = target_speed
        self.target_yaw = target_yaw

        balance_obs = self._get_move_obs()
        balance_obs_quant = [
            (np.round(obs_value / self.input_scale) + self.input_zero_point)
            for obs_value in balance_obs
        ]

        # need to make sure the quantized values are within the range of an int8, otherwise
        # the values get wrapped and -129 becomes +127! Which is obviously bad for the
        # robots balance
        balance_obs_quant = np.clip(balance_obs_quant, a_min = -128, a_max = 127)
        input_tensor = np.array([balance_obs_quant], dtype=np.int8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_data_quant = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        output_data = np.array(
            [
                self.output_scale * (output_data_quant[0].astype(np.float32) - self.output_zero_point),
                self.output_scale * (output_data_quant[1].astype(np.float32) - self.output_zero_point),
            ],
            dtype=np.float32
        )

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + output_data[0] * WHEEL_SPEED_DELTA_MAX
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + output_data[1] * WHEEL_SPEED_DELTA_MAX

        # print(self.data.joint('torso_l_wheel').qvel[0])
        self.data.actuator('motor_l_wheel').ctrl = [vel_l]
        self.data.actuator('motor_r_wheel').ctrl = [vel_r]

    def _correct_ray_dist_for_pitch(self, dist: float, hit_geom_id: int):
        # accounts for lidar hitting floor, and corrects distance for pitch
        # angle
        if dist > 0.3:
            # the real lidar has a range of 0.3m
            return 0.0, -1

        # * -1 because sim has pitch opposite to real robot
        pitch = -self.get_pitch()
        wheel_radius = 0.034
        height = 0.110
        floor_distance = wheel_radius / math.sin(pitch) + height / math.tan(pitch) - 0.010

        if dist >= floor_distance and floor_distance > 0:
            return 0.0, -1
        else:
            dist = dist * math.cos(pitch)
            return dist, hit_geom_id

    def get_ray_hit_and_dist(self):
        lidar_pos = self.data.body("front_indicator").xpos
        lidar_pos = np.array(lidar_pos, dtype=np.float64)
        geom_id = self.data.body("front_indicator").id

        rotation_matrix = self.data.body("front_indicator").xmat.reshape(3, 3)
        world_ray_directions = [np.dot(rotation_matrix, rd) for rd in self.ray_directions]
        world_ray_directions = np.array(world_ray_directions, np.float64)

        hit_geom_ids = np.zeros(8, dtype=np.int32)
        hit_geom_dists = np.zeros(8, dtype=np.float64)
        mujoco.mj_multiRay(
            m=self.model,
            d=self.data,
            pnt=lidar_pos,
            vec=world_ray_directions.flatten(),
            geomgroup=None,
            flg_static=1,
            bodyexclude=geom_id,
            geomid=hit_geom_ids,
            dist=hit_geom_dists,
            nray=8,
            cutoff=mujoco.mjMAXVAL
        )

        res_distances = []
        res_geom_ids = []
        for i in range(0, len(hit_geom_ids)):
            d = hit_geom_dists[i]
            g = hit_geom_ids[i]
            c_dist, c_geom_id = self._correct_ray_dist_for_pitch(dist=d,hit_geom_id=g)
            res_distances.append(c_dist)
            res_geom_ids.append(c_geom_id)

        # if distance is zero (no ray hits) set to max observable distance. Will make things easier for RL
        res_distances = [
            0.3 if d == 0.0 else d
            for d in res_distances
        ]

        # if distance is negative then something has penetrated so set back to zero
        res_distances = [
            0.0 if d < 0.0 else d
            for d in res_distances
        ]

        return res_distances, res_geom_ids

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
            if dt > 0.0:
                pitch_dot = (pitch - self.last_pitch) / dt

        self.last_time = ts
        self.last_pitch = pitch

        return pitch_dot

    def get_wheel_velocities(self) -> tuple[float, float]:
        vel_m_0 = self.data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = self.data.joint('torso_r_wheel').qvel[0]

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        return (vel_m_0, vel_m_1)

    def get_wheel_yaw(self) -> float:
        vel_l, vel_r = self.get_wheel_velocities()
        wheel_yaw = vel_l - (-1 * vel_r)
        return wheel_yaw

    def get_wheel_speed(self) -> float:
        vel_l, vel_r = self.get_wheel_velocities()
        wheel_speed = (vel_l + (-1 * vel_r)) / 2
        return wheel_speed

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

        return reward

    def _get_obs(self):
        # returns the observations needed for the model being trained/tested
        self.loop_count += 1

        n_wheel_speed = self.get_wheel_speed() / WHEEL_SPEED_MAX
        n_wheel_yaw = self.get_wheel_yaw() / YAW_MAX

        return np.array(
            [
                n_wheel_speed, n_wheel_yaw, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            dtype=np.float32
        ).ravel()

