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
from mani_skill.utils.scene_builder.ai2thor import ProcTHORSceneBuilder, ArchitecTHORSceneBuilder, iTHORSceneBuilder, RoboTHORSceneBuilder

from mani_skill.utils.scene_builder.table import TableSceneBuilder


@register_env("UprightStack-v1", max_episode_steps=200)
class UprightStackEnv(BaseEnv):
    """
    Task Description
    ----------------
    Upright + Stack(choose which one)

    Randomizations
    --------------
    Random Position 
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
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
        pose = sapien_utils.look_at([0.2, 0.8, 0.3], [0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _build_brick(self, length, width, height, color=[0.7, 0.2, 0.2, 1], density=2000):
        builder = self.scene.create_actor_builder()
        # 创建积木的材质
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(color)  # 默认红色积木（RGBA）
        mat.metallic = 0.0  # 非金属
        mat.roughness = 0.6  # 适度粗糙
        mat.specular = 0.3  # 适度反射
        # 碰撞体
        builder.add_box_collision(
            sapien.Pose(),  # 积木的中心位置
            [length / 2, width / 2, height / 2],  # 半尺寸
            density=density,  # 设定密度
        )
        # 可视化
        builder.add_box_visual(
            sapien.Pose(),  # 可视化中心位置
            [length / 2, width / 2, height / 2],  # 可视化的半尺寸
            material=mat,  # 采用积木材质
        )
        # 返回构建好的积木对象
        return builder.build(name="brick")

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA= actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[0.1, 0, 0.1]),
        )
        self.brick = self._build_brick(length=0.15, width=0.04, height=0.03, color=[random.random(), random.random(), random.random(), 1])

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
        """
        Generate random y positions with minimum separation distance
        
        Args:
            batch_size: Number of environments
            count: Number of positions to generate
            y_range: (min_y, max_y) tuple for position range
            min_distance: Minimum distance between positions
        """
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

            object_y = self._generate_separated_y_positions(
                b, 3, (-0.3, 0.3), 0.1
            )
            object_indices = torch.randperm(3, device=self.device)
            objects = [self.cubeA, self.brick, self.cubeB]
            object_rotations = [torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device)]
            for i, object_idx in enumerate(object_indices):
                object_xyz = torch.zeros((b, 3), device=self.device)
                object_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                object_xyz[..., 1] = object_y[:, object_idx]
                object_xyz[..., 2] = 0.02
                object_q = object_rotations[i].expand(b, 4)
                tool_pose = Pose.create_from_pq(p=object_xyz, q=object_q)
                objects[i].set_pose(tool_pose)

    def evaluate(self):
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeB_static = self.cubeB.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        success = is_cubeA_static * is_cubeB_static
        return {
            "is_cubeA_static": is_cubeA_static,
            "is_cubeB_static": is_cubeB_static,
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
@register_env("UprightStack-gen1", max_episode_steps=200)
class UprightStackEnv_gen1(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
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
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _build_brick(self, length, width, height, color=[0.7, 0.2, 0.2, 1], density=2000):
        builder = self.scene.create_actor_builder()
        # 创建积木的材质
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(color)  # 默认红色积木（RGBA）
        mat.metallic = 0.0  # 非金属
        mat.roughness = 0.6  # 适度粗糙
        mat.specular = 0.3  # 适度反射
        # 碰撞体
        builder.add_box_collision(
            sapien.Pose(),  # 积木的中心位置
            [length / 2, width / 2, height / 2],  # 半尺寸
            density=density,  # 设定密度
        )
        # 可视化
        builder.add_box_visual(
            sapien.Pose(),  # 可视化中心位置
            [length / 2, width / 2, height / 2],  # 可视化的半尺寸
            material=mat,  # 采用积木材质
        )
        # 返回构建好的积木对象
        return builder.build(name="brick")

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = iTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([0])
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'024_bowl'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name="cubeA")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'025_mug'}",
        )
        # builder = actors.get_actor_builder(
        #     self.scene,
        #     id=f"ycb:{'014_lemon'}",
        # )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="cubeB")
        self.brick = self._build_brick(length=0.15, width=0.04, height=0.03, color=[random.random(), random.random(), random.random(), 1])

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

            object_y = self._generate_separated_y_positions(
                b, 3, (-0.3, 0.3), 0.1
            )
            object_indices = torch.randperm(3, device=self.device)
            objects = [self.cubeA, self.brick, self.cubeB]
            object_rotations = [torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device)]
            for i, object_idx in enumerate(object_indices):
                object_xyz = torch.zeros((b, 3), device=self.device)
                object_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                object_xyz[..., 1] = object_y[:, object_idx]
                object_xyz[..., 2] = 0.02
                object_q = object_rotations[i].expand(b, 4)
                tool_pose = Pose.create_from_pq(p=object_xyz, q=object_q)
                objects[i].set_pose(tool_pose)

    def evaluate(self):
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeB_static = self.cubeB.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        success = is_cubeA_static * is_cubeB_static
        return {
            "is_cubeA_static": is_cubeA_static,
            "is_cubeB_static": is_cubeB_static,
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

@register_env("UprightStack-gen2", max_episode_steps=200)
class UprightStackEnv_gen2(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
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
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _build_brick(self, length, width, height, color=[0.7, 0.2, 0.2, 1], density=2000):
        builder = self.scene.create_actor_builder()
        # 创建积木的材质
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(color)  # 默认红色积木（RGBA）
        mat.metallic = 0.0  # 非金属
        mat.roughness = 0.6  # 适度粗糙
        mat.specular = 0.3  # 适度反射
        # 碰撞体
        builder.add_box_collision(
            sapien.Pose(),  # 积木的中心位置
            [length / 2, width / 2, height / 2],  # 半尺寸
            density=density,  # 设定密度
        )
        # 可视化
        builder.add_box_visual(
            sapien.Pose(),  # 可视化中心位置
            [length / 2, width / 2, height / 2],  # 可视化的半尺寸
            material=mat,  # 采用积木材质
        )
        # 返回构建好的积木对象
        return builder.build(name="brick")

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = iTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([8])
        # builder = actors.get_actor_builder(
        #     self.scene,
        #     id=f"ycb:{'014_lemon'}",
        # )
        # builder = actors.get_actor_builder(
        #     self.scene,
        #     id=f"ycb:{'025_mug'}",
        # )
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'077_rubiks_cube'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name="cubeA")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'015_peach'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="cubeB")
        self.brick = self._build_brick(length=0.15, width=0.04, height=0.03, color=[random.random(), random.random(), random.random(), 1])

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

            object_y = self._generate_separated_y_positions(
                b, 3, (-0.3, 0.3), 0.1
            )
            object_indices = torch.randperm(3, device=self.device)
            objects = [self.cubeA, self.brick, self.cubeB]
            object_rotations = [torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device)]
            for i, object_idx in enumerate(object_indices):
                object_xyz = torch.zeros((b, 3), device=self.device)
                object_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                object_xyz[..., 1] = object_y[:, object_idx]
                object_xyz[..., 2] = 0.02
                object_q = object_rotations[i].expand(b, 4)
                tool_pose = Pose.create_from_pq(p=object_xyz, q=object_q)
                objects[i].set_pose(tool_pose)

    def evaluate(self):
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeB_static = self.cubeB.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        success = is_cubeA_static * is_cubeB_static
        return {
            "is_cubeA_static": is_cubeA_static,
            "is_cubeB_static": is_cubeB_static,
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