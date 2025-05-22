from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

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
from mani_skill.utils.scene_builder.table import TableSceneBuilder_tool
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization

# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("ToolsTask-v1", max_episode_steps=200)
class ToolsTaskEnv(BaseEnv):
    """
    Task Description
    ----------------


    Randomizations
    --------------


    Success Conditions
    ------------------


    """

    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _base3_size = [2e-2, 1.5e-2, 1.5e-2]
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance =12e-4  # single side clearance
    _clearance_2 = 18e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size
    handle_length = 0.3
    branch_length = 0.1
    hook_length = 0.08
    side_length = 0.05
    width = 0.03
    height = 0.03
    radius = 0.05
    arm_reach = 0.35
    cube_half_size = 0.05

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         gpu_memory_config=GPUMemoryConfig(
    #             found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
    #         )
    #     )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.65, -0.75, 0.3], [0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 1920, 1088, 1, 0.01, 100)]
    
    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0.66, 0.66, 0.66, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")
    
    def _build_charger3(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        metal_mat = sapien.render.RenderMaterial()
        metal_mat.set_base_color([0.66, 0.66, 0.66, 1])
        metal_mat.metallic = 1.0
        metal_mat.roughness = 0.0
        metal_mat.specular = 1.0

        builder.add_box_collision(sapien.Pose([peg_size[0], gap, -peg_size[2]]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, -peg_size[2]]), peg_size, material=metal_mat
        )

        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, -peg_size[2]]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, -peg_size[2]]), peg_size, material=metal_mat
        )

        builder.add_box_collision(sapien.Pose([peg_size[0], 0, 2 * peg_size[2]]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], 0, 2 * peg_size[2]]), peg_size, material=metal_mat
        )

        base_mat = sapien.render.RenderMaterial()
        base_mat.set_base_color([1, 1, 1, 1])
        base_mat.metallic = 0.0
        base_mat.roughness = 0.1

        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=base_mat
        )

        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        
        return builder.build(name="charger3")


    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        # _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
        # _peg_gap = 7e-3  # charger peg gap
        # _clearance = 5e-4  # single side clearance
        # _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")
    
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
            sapien.Pose([handle_length - width / 2, hook_length / 2, 0]),
            [width / 2, hook_length / 2, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - width / 2, hook_length / 2, 0]),
            [width / 2, hook_length / 2, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")
    
    def _build_circular_tool(self, radius, handle_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0, 0, 1, 1])  
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_cylinder_collision(
            sapien.Pose([0, handle_length + 0.03, 0]),  
            radius=radius,
            half_length=height / 2,  
            density=500,
        )
        builder.add_cylinder_visual(
            sapien.Pose([0, handle_length + 0.03, 0]),
            radius=radius,
            half_length=height / 2,
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([0, -0.9 * radius + handle_length / 2 + 0.03, 0]),
            [width / 2, handle_length / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([0, -0.9 * radius + handle_length / 2 + 0.03, 0]),
            [width / 2, handle_length / 2, height / 2],
            material=mat,
        )

        return builder.build(name="circular_shaped_tool")
    
    def _build_y_shaped_tool(self, handle_length, branch_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0, 1, 0, 1]) 
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([0, handle_length / 2, 0]),
            [width / 2, handle_length / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([0, handle_length / 2, 0]),
            [width / 2, handle_length / 2, height / 2],
            material=mat,
        )

        branch_offset = handle_length / 2 - 2 * (branch_length / 2) * 0.707  
        branch_spacing = width / 2  

        builder.add_box_collision(
            sapien.Pose([-branch_length / 2 + branch_spacing, branch_offset + handle_length, 0], [0.3826834, 0, 0, 0.9238795]),
            [branch_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([-branch_length / 2 + branch_spacing, branch_offset + handle_length, 0], [0.3826834, 0, 0, 0.9238795]),
            [branch_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([branch_length / 2 - branch_spacing, branch_offset + handle_length, 0], [-0.3826834, 0, 0, 0.9238795]),
            [branch_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([branch_length / 2 - branch_spacing, branch_offset + handle_length, 0], [-0.3826834, 0, 0, 0.9238795]),
            [branch_length / 2, width / 2, height / 2],
            material=mat,
        )

        return builder.build(name="y_shaped_tool")


    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder_tool(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.charger3 = self._build_charger3(
            self._peg_size,
            self._base3_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance_2,
            ],
            self._receptacle_size,
            self._peg_gap,
        )
        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )
        self.circular_shape_tool = self._build_circular_tool(
            radius=self.radius,
            handle_length=self.handle_length * 3 / 4,
            width=self.width,
            height=self.height,
        )
        self.y_shape_tool = self._build_y_shaped_tool(
            handle_length=self.handle_length * 3 / 4,
            branch_length=self.branch_length,
            width=self.width,
            height=self.height,
        )

    # def _setup_sensors(self, options: dict):
    #     # default code here will setup all sensors. You can add additional code to change the sensors e.g.
    #     # if you want to randomize camera positions
    #     return super()._setup_sensors()

    # def _load_lighting(self, options: dict):
    #     for scene in self.scene.sub_scenes:
    #         scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
    #         scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
    #         scene.add_directional_light([0, 0, -1], [1, 1, 1])
        
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
            self.table_scene.initialize(env_idx)
            
            # Generate y positions for tools
            tool_y = self.generate_separated_y_positions(
                b, 3, (0.1, 0.4), 0.15
            )
            
            # Random permutation for tools left-to-right order
            tool_indices = torch.randperm(3, device=self.device)
            
            # Initialize tools
            tools = [self.circular_shape_tool, self.l_shape_tool, self.y_shape_tool]
            
            # Define rotations for each tool
            # Tool 1: 90° around z and 90° around x
            q1 = torch.tensor([-0.5, 0.5, 0.5, 0.5], device=self.device)  # Combined z and x rotation
            # Tool 2: No rotation
            q2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            # Tool 3: 90° around z
            q3 = torch.tensor([-0.7071068, 0.0, 0.0, 0.7071068], device=self.device)
            
            tool_rotations = [q1, q2, q3]
            
            for i, tool_idx in enumerate(tool_indices):
                tool_xyz = torch.zeros((b, 3), device=self.device)
                tool_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2  # x: [-0.1, 0.1)
                tool_xyz[..., 1] = tool_y[:, tool_idx]  # Use generated y positions
                tool_xyz[..., 2] = self.height / 2
                
                # Apply the corresponding rotation
                tool_q = tool_rotations[i].expand(b, 4)
                tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
                tools[i].set_pose(tool_pose)
            
            # Generate y positions for chargers
            charger_y = self.generate_separated_y_positions(
                b, 2, (-0.15, 0), 0.08
            )
            
            # Random permutation for chargers left-right order
            charger_indices = torch.randperm(2, device=self.device)
            
            # Initialize chargers
            chargers = [self.charger, self.charger3]
            for i, charger_idx in enumerate(charger_indices):
                charger_xyz = torch.zeros((b, 3), device=self.device)
                charger_xyz[..., 0] = (
                    self.arm_reach + 
                    0.02 * torch.rand(b, device=self.device) + self.handle_length - 0.4
                )
                charger_xyz[..., 1] = charger_y[:, charger_idx]
                charger_xyz[..., 2] = 0.05
                
                charger_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)
                charger_pose = Pose.create_from_pq(p=charger_xyz, q=charger_q)
                chargers[i].set_pose(charger_pose)

            xy = randomization.uniform([-0.25, -0.55], [0.0, -0.35], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = 0.2
            ori = randomization.random_quaternions(
                n=b,
                lock_x=True,
                lock_y=True,
                bounds=(3 * torch.pi / 4 - torch.pi / 8, 3 * torch.pi / 4 + torch.pi / 8),
            )
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )

            # print("agent_qpos:", self.agent.get_qpos())
            # self.agent.set_qpos(self.agent.get_qpos())

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))
    
    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    # def evaluate(self, obs: Any):
    def evaluate(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            success=success,
        )

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
