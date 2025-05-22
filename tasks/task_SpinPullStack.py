from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import random

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import articulations
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table import noTableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization
import os
from mani_skill.utils.building import ArticulationBuilder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.ai2thor import ProcTHORSceneBuilder, ArchitecTHORSceneBuilder, iTHORSceneBuilder, RoboTHORSceneBuilder

from mani_skill.utils.scene_builder.table import TableSceneBuilder

def create_rotating_disk(
    scene: sapien.Scene,
    rod_height=0.5,
    rod_radius=0.02,
    disk_radius=0.3,
    disk_thickness=0.01,
    joint_friction=0.0,
    joint_damping=0.1,
    density=1000.0,
) -> sapien.physx.PhysxArticulation:
    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    # ----------------------
    # 1. Create fixed rod (root)
    # ----------------------
    material = sapien.physx.PhysxMaterial(
        static_friction=0.6,  # Static friction coefficient
        dynamic_friction=0.5,  # Dynamic friction coefficient
        restitution=0.1,  # Restitution coefficient (elasticity)
    )

    rod: sapien.LinkBuilder = builder.create_link_builder()
    rod.set_name("rod")
    rod.add_capsule_collision(sapien.Pose(p=[0, 0, 0], q = euler2quat(0, 0, 0)), half_length=rod_height / 2, radius=rod_radius, density=density, material=material)
    rod.add_capsule_visual(sapien.Pose(p=[0, 0, 0], q = euler2quat(0, 0, 0)), half_length=rod_height / 2, radius=rod_radius, material=[0.2, 0.2, 0.8])

    # ----------------------
    # 2. Create rotating disk
    # ----------------------
    disk: sapien.LinkBuilder = builder.create_link_builder(rod)
    disk.set_name("disk")
    disk.set_joint_name("disk_joint")
    disk.add_cylinder_collision(sapien.Pose(p=[0, 0, 0], q = euler2quat(0, 0, 0)), half_length=disk_thickness / 2, radius=disk_radius, density=density, material=material)
    disk.add_cylinder_visual(sapien.Pose(p=[0, 0, 0], q = euler2quat(0, 0, 0)), half_length=disk_thickness / 2, radius=disk_radius, material=[0.8, 0.2, 0.2])

    # ----------------------
    # 3. Joint properties: revolute (rotational joint)
    # ----------------------
    disk.set_joint_properties(
        type="revolute",
        limits=[[-np.inf, np.inf]],  # Allow unlimited rotation
        pose_in_parent=sapien.Pose(
            p=[0, 0, 0],
            q=euler2quat(0, 0, 0),
        ),
        pose_in_child=sapien.Pose(
            p=[rod_height * 0.39, 0, 0],
            q=euler2quat(0, 0, 0),
        ),
        friction=joint_friction,
        damping=joint_damping,
    )

    # ----------------------
    # 4. Build articulation
    # ----------------------
    articulation = builder.build(name="disk", fix_root_link=True)
    return articulation

@register_env("SpinPullStack-v1", max_episode_steps=200)
class SpinPullStackEnv(BaseEnv):
    """
    Task Description
    ----------------
    Grasp one object(cube) on the rotated disk and stack on another object(cube). 

    Randomizations
    --------------
    Random Position 
    """

    SUPPORTED_ROBOTS = ["panda", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.4, -0.4, 0.3], [0.15, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 512, 512, 1.2, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # self.disk_model = MJCFLoader()
        # self.disk = self.disk_model.load('disk.xml')
        self.disk = create_rotating_disk(self.scene)
        self.disk.set_qpos([0.0])  

        self.cube= actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.cubeB= actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0.2, 0, 1, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
        with torch.device(self.device):
            y_positions = torch.zeros((batch_size, count), device=self.device)

            for b in range(batch_size):
                available_range = y_range[1] - y_range[0] - (count - 1) * min_distance
                if available_range < 0:
                    raise ValueError("Range too small for minimum distance requirement")
                
                positions = torch.rand(count, device=self.device) * available_range + y_range[0]
                positions = torch.sort(positions)[0]
                for i in range(1, count):
                    positions[i:] += min_distance
                y_positions[b] = positions
            return y_positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.disk_x = 0.25
            self.disk_y = -0.1
            self.disk.set_pose(sapien.Pose(p=[self.disk_x, self.disk_y, -0.12], q=euler2quat(0, np.pi / 2, 0)))
            self.disk.set_qpos(torch.tensor([0.0]))
            self.disk.set_qvel(torch.tensor([0.5]))

            cube_radius = self._generate_separated_y_positions(b, 2, (0.1, 0.28), 0.12)
            self.cube_r = cube_radius[:, 1]
            cube_angle = random.uniform(0, 2 * np.pi)
            cube_x = self.disk_x + np.cos(cube_angle) * self.cube_r
            cube_y = self.disk_y + np.sin(cube_angle) * self.cube_r
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = cube_x
            xyz[:, 1] = cube_y
            xyz[:, 2] = 0.12
            qs_cube = euler2quat(0, 0, 0)
            self.cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_cube))
            self.cubeB_r = cube_radius[:, 0]
            cubeB_angle = random.uniform(0, 2 * np.pi)
            cubeB_x = self.disk_x + np.cos(cubeB_angle) * self.cubeB_r
            cubeB_y = self.disk_y + np.sin(cubeB_angle) * self.cubeB_r
            xyzB = torch.zeros((b, 3))
            xyzB[:, 0] = cubeB_x
            xyzB[:, 1] = cubeB_y
            xyzB[:, 2] = 0.12
            qs_cubeB = euler2quat(0, 0, 0)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyzB.clone(), q=qs_cubeB))

    def step(self, action):
        action = self._step_action(action)
        self.disk.set_qvel(torch.tensor([1]))
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info)
        reward = self.get_reward(obs=obs, action=action, info=info)
        if "success" in info:

            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )

    def evaluate(self):
        is_cube_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        success = is_cube_static
        return {
            "is_cube_static": is_cube_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_state_dict(self):
        state = super().get_state_dict()
        return state

    def set_state_dict(self, state):
        self.goal_pos = state["goal_pos"]
        super().set_state_dict(state)

@register_env("SpinPullStack-gen1", max_episode_steps=200)
class SpinPullStackEnv_gen1(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.7, 0.3], [0.15, 0.0, 0.15]) # view 1(right)
        # pose = sapien_utils.look_at([1.2, -0.2, 0.3], [0.15, 0.0, 0.15])  # view 2(front)
        # pose = sapien_utils.look_at([0.3, 0.1, 1.1], [0.15, 0.0, 0.15])  # view 3(above)
        # pose = sapien_utils.look_at([-0.8, 0.6, 0.6], [0.15, 0.0, 0.15])  # view 4(left rear)
        return [CameraConfig("render_camera", pose, 512, 512, 1.2, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([5])
        self.disk = create_rotating_disk(self.scene)
        self.disk.set_qpos([0.0])  # 初始角度

        self.cube= actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0.8, 0.8, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'013_apple'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="apple")

        # self.cubeB= actors.build_cube(
        #     self.scene,
        #     half_size=0.02,
        #     color=[0.2, 0, 1, 1],
        #     name="cubeB",
        #     initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        # )

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
        with torch.device(self.device):
            y_positions = torch.zeros((batch_size, count), device=self.device)

            for b in range(batch_size):
                available_range = y_range[1] - y_range[0] - (count - 1) * min_distance
                if available_range < 0:
                    raise ValueError("Range too small for minimum distance requirement")
                
                positions = torch.rand(count, device=self.device) * available_range + y_range[0]
                positions = torch.sort(positions)[0]
                for i in range(1, count):
                    positions[i:] += min_distance
                y_positions[b] = positions
            return y_positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.disk_x = 0.25
            self.disk_y = -0.1
            self.disk.set_pose(sapien.Pose(p=[self.disk_x, self.disk_y, -0.12], q=euler2quat(0, np.pi / 2, 0)))
            self.disk.set_qpos(torch.tensor([0.0]))
            self.disk.set_qvel(torch.tensor([0.5]))

            cube_radius = self._generate_separated_y_positions(b, 2, (0.1, 0.28), 0.12)
            self.cube_r = cube_radius[:, 1]
            cube_angle = random.uniform(0, 2 * np.pi)
            cube_x = self.disk_x + np.cos(cube_angle) * self.cube_r
            cube_y = self.disk_y + np.sin(cube_angle) * self.cube_r
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = cube_x
            xyz[:, 1] = cube_y
            xyz[:, 2] = 0.12
            qs_cube = euler2quat(0, 0, 0)
            self.cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_cube))
            self.cubeB_r = cube_radius[:, 0]
            cubeB_angle = random.uniform(0, 2 * np.pi)
            cubeB_x = self.disk_x + np.cos(cubeB_angle) * self.cubeB_r
            cubeB_y = self.disk_y + np.sin(cubeB_angle) * self.cubeB_r
            xyzB = torch.zeros((b, 3))
            xyzB[:, 0] = cubeB_x
            xyzB[:, 1] = cubeB_y
            xyzB[:, 2] = 0.12
            qs_cubeB = euler2quat(0, 0, 0)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyzB.clone(), q=qs_cubeB))

    def step(self, action):
        action = self._step_action(action)
        self.disk.set_qvel(torch.tensor([1]))
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info)
        reward = self.get_reward(obs=obs, action=action, info=info)
        if "success" in info:

            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )

    def evaluate(self):
        is_cube_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        success = is_cube_static
        return {
            "is_cube_static": is_cube_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_state_dict(self):
        state = super().get_state_dict()
        return state

    def set_state_dict(self, state):
        self.goal_pos = state["goal_pos"]
        super().set_state_dict(state)

@register_env("SpinPullStack-gen2", max_episode_steps=200)
class SpinPullStackEnv_gen2(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.7, 0.3], [0.15, 0.0, 0.15]) # view 1(right)
        # pose = sapien_utils.look_at([1.2, 0, 0.3], [0.15, 0.0, 0.15])  # view 2(front)
        # pose = sapien_utils.look_at([0.3, 0.1, 1.1], [0.15, 0.0, 0.15])  # view 3(above)
        # pose = sapien_utils.look_at([-0.8, 0.6, 0.6], [0.15, 0.0, 0.15])  # view 4(left rear)
        return [CameraConfig("render_camera", pose, 512, 512, 1.2, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([5])
        self.disk = create_rotating_disk(self.scene)
        self.disk.set_qpos([0.0])  # 初始角度

        self.cube= actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0.3, 0.8, 0.4, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'024_bowl'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="apple")

        # self.cubeB= actors.build_cube(
        #     self.scene,
        #     half_size=0.02,
        #     color=[0.2, 0, 1, 1],
        #     name="cubeB",
        #     initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        # )

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
        with torch.device(self.device):
            y_positions = torch.zeros((batch_size, count), device=self.device)

            for b in range(batch_size):
                available_range = y_range[1] - y_range[0] - (count - 1) * min_distance
                if available_range < 0:
                    raise ValueError("Range too small for minimum distance requirement")
                
                positions = torch.rand(count, device=self.device) * available_range + y_range[0]
                positions = torch.sort(positions)[0]
                for i in range(1, count):
                    positions[i:] += min_distance
                y_positions[b] = positions
            return y_positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.disk_x = 0.25
            self.disk_y = -0.1
            self.disk.set_pose(sapien.Pose(p=[self.disk_x, self.disk_y, -0.12], q=euler2quat(0, np.pi / 2, 0)))
            self.disk.set_qpos(torch.tensor([0.0]))
            self.disk.set_qvel(torch.tensor([0.5]))

            cube_radius = self._generate_separated_y_positions(b, 2, (0.1, 0.28), 0.12)
            self.cube_r = cube_radius[:, 1]
            cube_angle = random.uniform(0, 2 * np.pi)
            cube_x = self.disk_x + np.cos(cube_angle) * self.cube_r
            cube_y = self.disk_y + np.sin(cube_angle) * self.cube_r
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = cube_x
            xyz[:, 1] = cube_y
            xyz[:, 2] = 0.12
            qs_cube = euler2quat(0, 0, 0)
            self.cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_cube))
            self.cubeB_r = cube_radius[:, 0]
            cubeB_angle = random.uniform(0, 2 * np.pi)
            cubeB_x = self.disk_x + np.cos(cubeB_angle) * self.cubeB_r
            cubeB_y = self.disk_y + np.sin(cubeB_angle) * self.cubeB_r
            xyzB = torch.zeros((b, 3))
            xyzB[:, 0] = cubeB_x
            xyzB[:, 1] = cubeB_y
            xyzB[:, 2] = 0.12
            qs_cubeB = euler2quat(0, 0, 0)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyzB.clone(), q=qs_cubeB))

    def step(self, action):
        action = self._step_action(action)
        self.disk.set_qvel(torch.tensor([1]))
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info)
        reward = self.get_reward(obs=obs, action=action, info=info)
        if "success" in info:

            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )

    def evaluate(self):
        is_cube_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        success = is_cube_static
        return {
            "is_cube_static": is_cube_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_state_dict(self):
        state = super().get_state_dict()
        return state

    def set_state_dict(self, state):
        self.goal_pos = state["goal_pos"]
        super().set_state_dict(state)