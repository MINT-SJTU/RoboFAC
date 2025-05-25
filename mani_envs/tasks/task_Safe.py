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
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import articulations
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization

from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.table import noTableSceneBuilder
from mani_skill.utils.scene_builder.ai2thor import ProcTHORSceneBuilder, ArchitecTHORSceneBuilder, iTHORSceneBuilder, RoboTHORSceneBuilder


@register_env("SafeTask-v1", max_episode_steps=200)
class SafeTaskEnv(BaseEnv):
    """
    Task Description
    ----------------
    1. Put the spoon in the cup
    2. move the cup out of the way
    3. pull the microwave door open
    4. put the cup in the microwave
    5. Close the microwave door

    Randomizations
    --------------
    Spoons and cups appear randomly within a given range of locations

    Success Conditions
    ------------------
    1. All objects are static.
    2. Nothing is being grasped in the gripper.
    3. The spoon is in the cup.
    4. The cup is in the microwave.
    5. The microwave door is closed.
    """

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
        pose = sapien_utils.look_at([0.4, 0.3, 0.2], [0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _build_gold_bar(self, length, width, height, top_width):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0.84, 0, 1])  
        mat.metallic = 1.0  
        mat.roughness = 0.1  
        mat.specular = 0.9  

        num_sections = 10  
        section_height = height / num_sections  

        for i in range(num_sections):
            current_width = width + (top_width - width) * (i / (num_sections - 1))
            
            builder.add_box_collision(
                sapien.Pose([0, 0, section_height * (i + 0.5)]),  
                [length / 2, current_width / 2, section_height / 2],  
                density=8000,  
            )
            
            builder.add_box_visual(
                sapien.Pose([0, 0, section_height * (i + 0.5)]),  
                [length / 2, current_width / 2, section_height / 2],  
                material=mat, 
            )

        return builder.build(name="gold_bar")



    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{101593}"
        )
        builder.inital_pose = sapien.Pose(p=[2, 2, 5])
        self.safe = builder.build(name="safe")
        self.goldbar = self._build_gold_bar(length = 0.12, width = 0.036, height = 0.025, top_width = 0.03)

    # def _setup_sensors(self, options: dict):
    #     # default code here will setup all sensors. You can add additional code to change the sensors e.g.
    #     # if you want to randomize camera positions
    #     return super()._setup_sensors()

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02  
          
            safe_region = [[-0.05, -0.45], [0.05, -0.25]]
            safe_sampler = randomization.UniformPlacementSampler(
                bounds=safe_region, batch_size=b, device=self.device
            )
           
            goldbar_region = [[0.1, 0.2], [-0.25, 0.1]]
            goldbar_sampler = randomization.UniformPlacementSampler(
                bounds=goldbar_region, batch_size=b, device=self.device
            )
            safe_xy = safe_sampler.sample(0.2, 100)
            xyz[:, :2] = safe_xy
            xyz[:, 2] = 0.14
            qs_safe = euler2quat(0, 0, - np.pi / 2)  
            self.safe.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_safe))
            self.safe.set_qpos(torch.tensor([np.pi * 0.6, np.pi * 0.25, 0.0]))

            goldbar_xy = goldbar_sampler.sample(0.2, 100)
            xyz[:, :2] = goldbar_xy
            xyz[:, 2] = 0.02
            qs_goldbar = euler2quat(0, 0, np.pi * 0.5)  
            self.goldbar.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_goldbar))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = (self.goldbar.is_static(lin_thresh=1e-2, ang_thresh=0.5)) * (abs(self.safe.get_qvel()[0, 0]) < 0.1) * (abs(self.safe.get_qvel()[0, 1]) < 0.1)* (abs(self.safe.get_qvel()[0, 2]) < 0.1)
        is_safe_close = torch.tensor([abs(self.safe.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.goldbar)
        is_knob_rotation_correct = torch.tensor([abs(self.safe.get_qpos()[0, 1]) > 3.6])
        pos_goldbar = self.goldbar.pose.p
        pos_safe = self.safe.get_pose().p
        offset_goldbar_safe = pos_goldbar - pos_safe
        is_goldbar_in_safe = ( torch.linalg.norm(offset_goldbar_safe[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_goldbar_safe[..., 2]) < 0.13) * (torch.abs(offset_goldbar_safe[..., 2]) > 0.08)
        success = is_all_static * is_safe_close * (~is_something_grasped) * is_knob_rotation_correct * is_goldbar_in_safe
        # print(f'\nis_all_static: {is_all_static}')
        # print(f'is_safe_close: {is_safe_close}')
        # print(f'is_something_grasped: {is_something_grasped}')
        # print(f'is_knob_rotation_correct: {is_knob_rotation_correct}')
        # print(f'is_goldbar_in_safe: {is_goldbar_in_safe}')
        # print(f'knob angle: {abs(self.safe.get_qpos()[0, 1])}')
        # print(f'horizontal offset: {torch.linalg.norm(offset_goldbar_safe[..., :2], axis=1)}')
        # print(f'vertical offset: {torch.abs(offset_goldbar_safe[..., 2])}')
        # print(f'success: {success}')
        return {
            "is_all_static": is_all_static,
            "is_safe_close": is_safe_close,
            "is_something_grasped": is_something_grasped,
            "is_knob_rotation_correct": is_knob_rotation_correct,
            "is_goldbar_in_safe": is_goldbar_in_safe,
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

@register_env("SafeTask-usb", max_episode_steps=200)
class SafeTaskEnv_usb(BaseEnv):

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
        # pose = sapien_utils.look_at([0.7, 0.5, 0.2], [0.0, 0.0, 0.15]) # (5)
        pose = sapien_utils.look_at([0.6, -0.5, 0.5], [0.0, 0.0, 0.15]) # (4)
        # pose = sapien_utils.look_at([0.3, 0.2, 0.8], [0.0, 0.0, 0.15]) # (3)
        # pose = sapien_utils.look_at([-0.6, 0.5, 0.4], [0.0, 0.0, 0.15])  # (1)
        # pose = sapien_utils.look_at([0.2, 0.9, 0.1], [0.0, 0.0, 0.15]) # (2) keng
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _build_usb_drive(self, length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0.2, 0.2, 0.2, 1])  
        mat.metallic = 0.1  
        mat.roughness = 0.5  
        mat.specular = 0.6  

        builder.add_box_collision(sapien.Pose(), [length / 2, width / 2, height / 2], density=2000)
        builder.add_box_visual(sapien.Pose(), [length / 2, width / 2, height / 2], material=mat)

        usb_mat = sapien.render.RenderMaterial()
        usb_mat.set_base_color([0.7, 0.7, 0.7, 1])  
        usb_mat.metallic = 1.0
        usb_mat.roughness = 0.2
        usb_mat.specular = 0.9

        builder.add_box_visual(sapien.Pose([length / 2, 0, 0]), [length / 4, width / 3, height / 3], material=usb_mat)

        return builder.build(name="usb_drive")


    def _load_scene(self, options: dict):
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.replicaCAD_scene = ReplicaCADSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.replicaCAD_scene.build(1)
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{101593}"
        )
        builder.inital_pose = sapien.Pose(p=[2, 2, 5])
        self.safe = builder.build(name="safe")
        self.goldbar = self._build_usb_drive(length=0.08, width=0.036, height=0.02)

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02  
      
            safe_region = [[-0.05, -0.45], [0.05, -0.25]]
            safe_sampler = randomization.UniformPlacementSampler(
                bounds=safe_region, batch_size=b, device=self.device
            )
           
            goldbar_region = [[0.1, 0.2], [-0.25, 0.1]]
            goldbar_sampler = randomization.UniformPlacementSampler(
                bounds=goldbar_region, batch_size=b, device=self.device
            )
    
            safe_xy = safe_sampler.sample(0.2, 100)
            xyz[:, :2] = safe_xy
            xyz[:, 2] = 0.14
   
            qs_safe = euler2quat(0, 0, - np.pi / 2) 
            self.safe.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_safe))
            self.safe.set_qpos(torch.tensor([np.pi * 0.6, np.pi * 0.25, 0.0]))


            goldbar_xy = goldbar_sampler.sample(0.2, 100)
            xyz[:, :2] = goldbar_xy
            xyz[:, 2] = 0.02
        
            qs_goldbar = euler2quat(0, 0, - np.pi * 0.5) 
            self.goldbar.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_goldbar))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = (self.goldbar.is_static(lin_thresh=1e-2, ang_thresh=0.5)) * (abs(self.safe.get_qvel()[0, 0]) < 0.1) * (abs(self.safe.get_qvel()[0, 1]) < 0.1)* (abs(self.safe.get_qvel()[0, 2]) < 0.1)
        is_safe_close = torch.tensor([abs(self.safe.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.goldbar)
        is_knob_rotation_correct = torch.tensor([abs(self.safe.get_qpos()[0, 1]) > 3.6])
        pos_goldbar = self.goldbar.pose.p
        pos_safe = self.safe.get_pose().p
        offset_goldbar_safe = pos_goldbar - pos_safe
        is_goldbar_in_safe = ( torch.linalg.norm(offset_goldbar_safe[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_goldbar_safe[..., 2]) < 0.13) * (torch.abs(offset_goldbar_safe[..., 2]) > 0.08)
        success = is_all_static * is_safe_close * (~is_something_grasped) * is_knob_rotation_correct * is_goldbar_in_safe
        return {
            "is_all_static": is_all_static,
            "is_safe_close": is_safe_close,
            "is_something_grasped": is_something_grasped,
            "is_knob_rotation_correct": is_knob_rotation_correct,
            "is_goldbar_in_safe": is_goldbar_in_safe,
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

@register_env("SafeTask-screwdriver", max_episode_steps=200)
class SafeTaskEnv_screwdriver(BaseEnv):

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
        # pose = sapien_utils.look_at([0.7, 0.5, 0.2], [0.0, 0.0, 0.15]) # (4)
        # pose = sapien_utils.look_at([0.6, -0.5, 0.5], [0.0, 0.0, 0.15]) # (3)
        # pose = sapien_utils.look_at([0.3, 0.2, 0.8], [0.0, 0.0, 0.15]) # (2)
        pose = sapien_utils.look_at([-0.6, 0.5, 0.4], [0.0, 0.0, 0.15]) # (5)
        # pose = sapien_utils.look_at([0.2, 0.9, 0.1], [0.0, 0.0, 0.15]) # (1)
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([2])
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{101593}"
        )
        builder.inital_pose = sapien.Pose(p=[2, 2, 5])
        self.safe = builder.build(name="safe")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'043_phillips_screwdriver'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.goldbar = builder.build(name="goldbar")

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02  
          
            safe_region = [[-0.05, -0.45], [0.05, -0.25]]
            safe_sampler = randomization.UniformPlacementSampler(
                bounds=safe_region, batch_size=b, device=self.device
            )
         
            goldbar_region = [[0.1, 0.2], [-0.25, 0.1]]
            goldbar_sampler = randomization.UniformPlacementSampler(
                bounds=goldbar_region, batch_size=b, device=self.device
            )
          
            safe_xy = safe_sampler.sample(0.2, 100)
            xyz[:, :2] = safe_xy
            xyz[:, 2] = 0.14
       
            qs_safe = euler2quat(0, 0, - np.pi / 2) 
            self.safe.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_safe))
            self.safe.set_qpos(torch.tensor([np.pi * 0.6, np.pi * 0.25, 0.0]))

       
            goldbar_xy = goldbar_sampler.sample(0.2, 100)
            xyz[:, :2] = goldbar_xy
            xyz[:, 2] = 0.02
          
            qs_goldbar = euler2quat(0, 0, np.pi * 0.5)  
            self.goldbar.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_goldbar))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = (self.goldbar.is_static(lin_thresh=1e-2, ang_thresh=0.5)) * (abs(self.safe.get_qvel()[0, 0]) < 0.1) * (abs(self.safe.get_qvel()[0, 1]) < 0.1)* (abs(self.safe.get_qvel()[0, 2]) < 0.1)
        is_safe_close = torch.tensor([abs(self.safe.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.goldbar)
        is_knob_rotation_correct = torch.tensor([abs(self.safe.get_qpos()[0, 1]) > 3.6])
        pos_goldbar = self.goldbar.pose.p
        pos_safe = self.safe.get_pose().p
        offset_goldbar_safe = pos_goldbar - pos_safe
        is_goldbar_in_safe = ( torch.linalg.norm(offset_goldbar_safe[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_goldbar_safe[..., 2]) < 0.13) * (torch.abs(offset_goldbar_safe[..., 2]) > 0.08)
        success = is_all_static * is_safe_close * (~is_something_grasped) * is_knob_rotation_correct * is_goldbar_in_safe
        return {
            "is_all_static": is_all_static,
            "is_safe_close": is_safe_close,
            "is_something_grasped": is_something_grasped,
            "is_knob_rotation_correct": is_knob_rotation_correct,
            "is_goldbar_in_safe": is_goldbar_in_safe,
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

@register_env("SafeTask-hammer", max_episode_steps=200)
class SafeTaskEnv_hammer(BaseEnv):

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
        # pose = sapien_utils.look_at([0.7, 0.5, 0.2], [0.0, 0.0, 0.15]) # (5)
        # pose = sapien_utils.look_at([0.6, -0.5, 0.5], [0.0, 0.0, 0.15]) # (3)
        # pose = sapien_utils.look_at([0.3, 0.2, 0.8], [0.0, 0.0, 0.15]) # (2)
        # pose = sapien_utils.look_at([-0.6, 0.5, 0.4], [0.0, 0.0, 0.15]) # (1)
        pose = sapien_utils.look_at([0.2, 0.9, 0.1], [0.0, 0.0, 0.15]) # (4)
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([5])
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{101593}"
        )
        builder.inital_pose = sapien.Pose(p=[2, 2, 5])
        self.safe = builder.build(name="safe")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'048_hammer'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.goldbar = builder.build(name="goldbar")

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02  
            safe_region = [[-0.05, -0.45], [0.05, -0.25]]
            safe_sampler = randomization.UniformPlacementSampler(
                bounds=safe_region, batch_size=b, device=self.device
            )
            goldbar_region = [[0.1, 0.2], [-0.25, 0.1]]
            goldbar_sampler = randomization.UniformPlacementSampler(
                bounds=goldbar_region, batch_size=b, device=self.device
            )
            safe_xy = safe_sampler.sample(0.2, 100)
            xyz[:, :2] = safe_xy
            xyz[:, 2] = 0.14
            qs_safe = euler2quat(0, 0, - np.pi / 2)  
            self.safe.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_safe))
            self.safe.set_qpos(torch.tensor([np.pi * 0.6, np.pi * 0.25, 0.0]))

            goldbar_xy = goldbar_sampler.sample(0.2, 100)
            xyz[:, :2] = goldbar_xy
            xyz[:, 2] = 0.02
            qs_goldbar = euler2quat(0, 0, np.pi)  
            self.goldbar.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_goldbar))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = (self.goldbar.is_static(lin_thresh=1e-2, ang_thresh=0.5)) * (abs(self.safe.get_qvel()[0, 0]) < 0.1) * (abs(self.safe.get_qvel()[0, 1]) < 0.1)* (abs(self.safe.get_qvel()[0, 2]) < 0.1)
        is_safe_close = torch.tensor([abs(self.safe.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.goldbar)
        is_knob_rotation_correct = torch.tensor([abs(self.safe.get_qpos()[0, 1]) > 3.6])
        pos_goldbar = self.goldbar.pose.p
        pos_safe = self.safe.get_pose().p
        offset_goldbar_safe = pos_goldbar - pos_safe
        is_goldbar_in_safe = ( torch.linalg.norm(offset_goldbar_safe[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_goldbar_safe[..., 2]) < 0.13) * (torch.abs(offset_goldbar_safe[..., 2]) > 0.08)
        success = is_all_static * is_safe_close * (~is_something_grasped) * is_knob_rotation_correct * is_goldbar_in_safe
        return {
            "is_all_static": is_all_static,
            "is_safe_close": is_safe_close,
            "is_something_grasped": is_something_grasped,
            "is_knob_rotation_correct": is_knob_rotation_correct,
            "is_goldbar_in_safe": is_goldbar_in_safe,
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

@register_env("SafeTask-spatula", max_episode_steps=200)
class SafeTaskEnv_spatula(BaseEnv):

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
        # pose = sapien_utils.look_at([0.7, 0.5, 0.2], [0.0, 0.0, 0.15]) # (3)
        # pose = sapien_utils.look_at([0.6, -0.5, 0.5], [0.0, 0.0, 0.15])  # (2)
        pose = sapien_utils.look_at([0.3, 0.2, 0.8], [0.0, 0.0, 0.15]) # (1)
        # pose = sapien_utils.look_at([-0.6, 0.5, 0.4], [0.0, 0.0, 0.15]) # (5)
        # pose = sapien_utils.look_at([0.2, 0.9, 0.1], [0.0, 0.0, 0.15]) # (4)
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = noTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([5])
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{101593}"
        )
        builder.inital_pose = sapien.Pose(p=[2, 2, 5])
        self.safe = builder.build(name="safe")
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'033_spatula'}",
        )
        builder.inital_pose = sapien.Pose(p=[0, 0, 0.5])
        self.goldbar = builder.build(name="goldbar")

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02  
            safe_region = [[-0.05, -0.45], [0.05, -0.25]]
            safe_sampler = randomization.UniformPlacementSampler(
                bounds=safe_region, batch_size=b, device=self.device
            )
            goldbar_region = [[0.1, 0.2], [-0.25, 0.1]]
            goldbar_sampler = randomization.UniformPlacementSampler(
                bounds=goldbar_region, batch_size=b, device=self.device
            )
            safe_xy = safe_sampler.sample(0.2, 100)
            xyz[:, :2] = safe_xy
            xyz[:, 2] = 0.14
            qs_safe = euler2quat(0, 0, - np.pi / 2)  
            self.safe.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_safe))
            self.safe.set_qpos(torch.tensor([np.pi * 0.6, np.pi * 0.25, 0.0]))

            goldbar_xy = goldbar_sampler.sample(0.2, 100)
            xyz[:, :2] = goldbar_xy
            xyz[:, 2] = 0.02
            # 
            qs_goldbar = euler2quat(0, 0, - np.pi / 2)  
            self.goldbar.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs_goldbar))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = (self.goldbar.is_static(lin_thresh=1e-2, ang_thresh=0.5)) * (abs(self.safe.get_qvel()[0, 0]) < 0.1) * (abs(self.safe.get_qvel()[0, 1]) < 0.1)* (abs(self.safe.get_qvel()[0, 2]) < 0.1)
        is_safe_close = torch.tensor([abs(self.safe.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.goldbar)
        is_knob_rotation_correct = torch.tensor([abs(self.safe.get_qpos()[0, 1]) > 3.6])
        pos_goldbar = self.goldbar.pose.p
        pos_safe = self.safe.get_pose().p
        offset_goldbar_safe = pos_goldbar - pos_safe
        is_goldbar_in_safe = ( torch.linalg.norm(offset_goldbar_safe[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_goldbar_safe[..., 2]) < 0.13) * (torch.abs(offset_goldbar_safe[..., 2]) > 0.08)
        success = is_all_static * is_safe_close * (~is_something_grasped) * is_knob_rotation_correct * is_goldbar_in_safe
        return {
            "is_all_static": is_all_static,
            "is_safe_close": is_safe_close,
            "is_something_grasped": is_something_grasped,
            "is_knob_rotation_correct": is_knob_rotation_correct,
            "is_goldbar_in_safe": is_goldbar_in_safe,
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