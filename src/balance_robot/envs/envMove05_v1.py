import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from balance_robot.envs.RobotMoveBaseEnv import (
    RobotMoveBaseEnv,
    WHEEL_SPEED_DELTA_MAX,
    YAW_MAX,
    WHEEL_SPEED_MAX
)


class EnvMove05(RobotMoveBaseEnv):

    def __init__(self, **kwargs):
        RobotMoveBaseEnv.__init__(self, 'envMove05_v1.xml', **kwargs)

        self.target_vel = 20.0


    def step(self, a):
        reward = self._get_reward()

        self._step_wheel_speeds(a[0] * 20.0, a[1] * YAW_MAX)
        # self._step_wheel_speeds(15, 0)


        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)


        self._update_camera_follow()

        # dists, hits = self.get_ray_hit_and_dist()
        # print("get_ray_hit_and_dist")
        # print(dists)
        # print(hits)



        # lidar_pos = self.data.body("front_indicator").xpos
        # lidar_pos = np.array(lidar_pos, dtype=np.float64)
        # geom_id = self.data.body("front_indicator").id

        # ray_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # rotation_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # angles =  np.arange(-50, 50.1, 14.285) * (math.pi / 180)
        # ray_directions = []
        # for angle in angles:
        #     rotation = Rotation.from_rotvec(angle * rotation_axis)
        #     rotated_vector = rotation.apply(ray_direction)
        #     ray_directions.append(rotated_vector)
        # rotation_matrix = self.data.body("front_indicator").xmat.reshape(3, 3)
        # world_ray_directions = [np.dot(rotation_matrix, rd) for rd in ray_directions]
        # world_ray_directions = np.array(world_ray_directions, np.float64)
        # # print(world_ray_directions)
        # hit_geom_ids = np.zeros(8, dtype=np.int32)
        # hit_geom_dists = np.zeros(8, dtype=np.float64)
        # mujoco.mj_multiRay(
        #     m=self.model,
        #     d=self.data,
        #     pnt=lidar_pos,
        #     vec=world_ray_directions.flatten(),
        #     geomgroup=None,
        #     flg_static=1,
        #     bodyexclude=geom_id,
        #     geomid=hit_geom_ids,
        #     dist=hit_geom_dists,
        #     nray=8,
        #     cutoff=mujoco.mjMAXVAL
        # )
        # print(hit_geom_ids)

        # rotation_matrix = self.data.body("front_indicator").xmat.reshape(3, 3)
        # ray_direction = np.dot(rotation_matrix, ray_direction)

        # hit_geom_id = np.zeros(1, dtype=np.int32)
        # # mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, unused)
        # dist = mujoco.mj_ray(
        #     self.model, self.data, lidar_pos, ray_direction, None, 1, geom_id, hit_geom_id
        # )
        # if (hit_geom_id[0] == -1):
        #     print("nothing")
        # else:
        #     print(hit_geom_id[0])
        #     print(self.model.geom(hit_geom_id[0]).name)


        # terminate if pitch is greater than 50deg
        terminated = np.abs(self.get_pitch()) > (50 * math.pi / 180)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # print(ob)
        # print(reward)
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}


    def _get_reward(self):
        reward = 0.5

        pitch = self.get_pitch()
        wheel_speed = self.get_wheel_speed()
        dv = self.target_wheel_speed - wheel_speed

        reward += 0.03 * (wheel_speed - self.target_wheel_speed) / self.target_wheel_speed

        dists, _ = self.get_ray_hit_and_dist()
        for d_index in range(2,6):
            d = dists[d_index]
            reward += 0.15 * (d - 0.15) / 0.3
        return reward


    def reset_model(self):
        self.target_wheel_speed = self.np_random.uniform(low=1, high=10)
        self.target_wheel_speed += 30

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

