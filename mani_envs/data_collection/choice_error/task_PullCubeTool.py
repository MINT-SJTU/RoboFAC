from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import noTableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
# from mani_skill.utils.scene_builder.robocasa import RoboCasaSceneBuilder
from mani_skill.utils.scene_builder.ai2thor import ProcTHORSceneBuilder, ArchitecTHORSceneBuilder, iTHORSceneBuilder, RoboTHORSceneBuilder

"""
**Task Description**
Given an L-shaped tool that is within the reach of the robot, leverage the
tool to pull a cube that is out of it's reach

**Randomizations**
- The cube's position (x,y) is randomized on top of a table in the region "<out of manipulator
reach, but within reach of tool>". It is placed flat on the table
- The target goal region is the region on top of the table marked by "<within reach of arm>"

**Success Conditions**
- The cube's xy position is within the goal region of the arm's base (marked by reachability)
"""

@register_env("PullCubeTool-golf-more", max_episode_steps=100)
class PullCubeToolEnv_golf_more(BaseEnv):
    """
    Replace with pulling a golf ball, use ReplicaCAD (seed 1), and apply randomized lighting.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.25
    hook_length = 0.05
    width = 0.04
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=512,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.8, -0.6, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=1440,
                height=1440,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = noTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.replicaCAD_scene = ReplicaCADSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.replicaCAD_scene.build(1)

        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        #     body_type="dynamic",
        # )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'058_golf_ball'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name="cube")        

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'073-b_lego_duplo'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name="cubeA")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'007_tuna_fish_can'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="cubeB")  

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
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
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            cube_y = self.generate_separated_y_positions(
                b, 3, (-0.25, 0.25), 0.1
            )

            cube_indices = torch.randperm(3, device=self.device)
            
            cubes = [self.cube, self.cubeA, self.cubeB]

            cube_q1 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q2 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q3 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            
            cube_rotations = [cube_q1, cube_q2, cube_q3]
            
            for i, cube_idx in enumerate(cube_indices):
                cube_xyz = torch.zeros((b, 3), device=self.device)
                # cube_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                cube_xyz[..., 0] = (self.arm_reach + torch.rand(b, device=self.device) * (self.handle_length) - 0.32)
                cube_xyz[..., 1] = cube_y[:, cube_idx]
                cube_xyz[..., 2] = self.cube_size / 2 + 0.015
                cube_q = cube_rotations[i].expand(b, 4)
                cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
                cubes[i].set_pose(cube_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tool_pose=self.l_shape_tool.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        # Success condition - cube is pulled close enough
        cube_pulled_close = cube_to_base_dist < 0.6

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
    

@register_env("PullCubeTool-dice-more", max_episode_steps=100)
class PullCubeToolEnv_dice_more(BaseEnv):
    """
    Replace with pulling a dice, use ReplicaCAD (seed 2), and apply randomized lighting.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.25
    hook_length = 0.05
    width = 0.04
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.5, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=1440,
                height=1440,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = noTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.replicaCAD_scene = ReplicaCADSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.replicaCAD_scene.build(1)

        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        #     body_type="dynamic",
        # )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'062_dice'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name="cube")        

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'012_strawberry'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name="cubeA")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'077_rubiks_cube'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="cubeB")  

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
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
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            cube_y = self.generate_separated_y_positions(
                b, 3, (-0.25, 0.25), 0.1
            )

            cube_indices = torch.randperm(3, device=self.device)
            
            cubes = [self.cube, self.cubeA, self.cubeB]

            cube_q1 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q2 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q3 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            
            cube_rotations = [cube_q1, cube_q2, cube_q3]
            
            for i, cube_idx in enumerate(cube_indices):
                cube_xyz = torch.zeros((b, 3), device=self.device)
                # cube_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                cube_xyz[..., 0] = (self.arm_reach + torch.rand(b, device=self.device) * (self.handle_length) - 0.32)
                cube_xyz[..., 1] = cube_y[:, cube_idx]
                cube_xyz[..., 2] = self.cube_size / 2 + 0.015
                cube_q = cube_rotations[i].expand(b, 4)
                cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
                cubes[i].set_pose(cube_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tool_pose=self.l_shape_tool.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        # Success condition - cube is pulled close enough
        cube_pulled_close = cube_to_base_dist < 0.6

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward


@register_env("PullCubeTool-can-more", max_episode_steps=100)
class PullCubeToolEnv_can_more(BaseEnv):
    """
    Replace with pulling a tomato soup can, use ReplicaCAD (seed 4), and apply randomized lighting.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.25
    hook_length = 0.05
    width = 0.04
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.9, -0.8, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=1440,
                height=1440,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = noTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.replicaCAD_scene = ReplicaCADSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.replicaCAD_scene.build(4)

        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        #     body_type="dynamic",
        # )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'005_tomato_soup_can'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name="cube")        

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'013_apple'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name="cubeA")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'065-a_cups'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="cubeB")  

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
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
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            cube_y = self.generate_separated_y_positions(
                b, 3, (-0.3, 0.1), 0.1
            )

            cube_indices = torch.randperm(3, device=self.device)
            
            cubes = [self.cube, self.cubeA, self.cubeB]

            cube_q1 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q2 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q3 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            
            cube_rotations = [cube_q1, cube_q2, cube_q3]
            
            for i, cube_idx in enumerate(cube_indices):
                cube_xyz = torch.zeros((b, 3), device=self.device)
                # cube_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                cube_xyz[..., 0] = (self.arm_reach + torch.rand(b, device=self.device) * (self.handle_length) - 0.35)
                cube_xyz[..., 1] = cube_y[:, cube_idx]
                cube_xyz[..., 2] = self.cube_size / 2 + 0.015
                cube_q = cube_rotations[i].expand(b, 4)
                cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
                cubes[i].set_pose(cube_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tool_pose=self.l_shape_tool.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        # Success condition - cube is pulled close enough
        cube_pulled_close = cube_to_base_dist < 0.6

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
    

@register_env("PullCubeTool-peach-more", max_episode_steps=100)
class PullCubeToolEnv_peach_more(BaseEnv):
    """
    Replace with pulling a peach, use ArchitecTHORSceneBuilder (seed [2]), and apply randomized lighting.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.25
    hook_length = 0.05
    width = 0.04
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.5, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=1440,
                height=1440,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = noTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([2])

        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        #     body_type="dynamic",
        # )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'015_peach'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name="cube")        

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'017_orange'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name="cubeA")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'056_tennis_ball'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="cubeB")  

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
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
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            cube_y = self.generate_separated_y_positions(
                b, 3, (-0.25, 0.25), 0.1
            )

            cube_indices = torch.randperm(3, device=self.device)
            
            cubes = [self.cube, self.cubeA, self.cubeB]

            cube_q1 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q2 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q3 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            
            cube_rotations = [cube_q1, cube_q2, cube_q3]
            
            for i, cube_idx in enumerate(cube_indices):
                cube_xyz = torch.zeros((b, 3), device=self.device)
                # cube_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                cube_xyz[..., 0] = (self.arm_reach + torch.rand(b, device=self.device) * (self.handle_length) - 0.32)
                cube_xyz[..., 1] = cube_y[:, cube_idx]
                cube_xyz[..., 2] = self.cube_size / 2 + 0.015
                cube_q = cube_rotations[i].expand(b, 4)
                cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
                cubes[i].set_pose(cube_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tool_pose=self.l_shape_tool.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        # Success condition - cube is pulled close enough
        cube_pulled_close = cube_to_base_dist < 0.6

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
    

@register_env("PullCubeTool-marble-more", max_episode_steps=100)
class PullCubeToolEnv_marble_more(BaseEnv):
    """
    Replace with pulling a (possibly marble) ball, use ArchitecTHORSceneBuilder (seed [5]), and apply randomized lighting.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.25
    hook_length = 0.05
    width = 0.04
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.5, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=1440,
                height=1440,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = noTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([5])

        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        #     body_type="dynamic",
        # )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'063-b_marbles'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name="cube")        

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'018_plum'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name="cubeA")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'073-d_lego_duplo'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name="cubeB")  

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
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
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            cube_y = self.generate_separated_y_positions(
                b, 3, (-0.25, 0.25), 0.1
            )

            cube_indices = torch.randperm(3, device=self.device)
            
            cubes = [self.cube, self.cubeA, self.cubeB]

            cube_q1 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q2 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            cube_q3 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device,)
            
            cube_rotations = [cube_q1, cube_q2, cube_q3]
            
            for i, cube_idx in enumerate(cube_indices):
                cube_xyz = torch.zeros((b, 3), device=self.device)
                # cube_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                cube_xyz[..., 0] = (self.arm_reach + torch.rand(b, device=self.device) * (self.handle_length) - 0.32)
                cube_xyz[..., 1] = cube_y[:, cube_idx]
                cube_xyz[..., 2] = self.cube_size / 2 + 0.015
                cube_q = cube_rotations[i].expand(b, 4)
                cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
                cubes[i].set_pose(cube_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tool_pose=self.l_shape_tool.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        # Success condition - cube is pulled close enough
        cube_pulled_close = cube_to_base_dist < 0.6

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward