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
from mani_skill.utils.scene_builder.table import noTableSceneBuilder_microwave
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization

from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.ai2thor import ProcTHORSceneBuilder, ArchitecTHORSceneBuilder, iTHORSceneBuilder, RoboTHORSceneBuilder

from mani_skill.utils.scene_builder.table import TableSceneBuilder


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("MicrowaveTask-v1", max_episode_steps=200)
class MicrowaveTaskEnv(BaseEnv):
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
        pose = sapien_utils.look_at([0.4, -0., 0.1], [0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # 加载微波炉
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{7167}"
        )
        builder.inital_pose = sapien.Pose(p=[10, 10, 10])
        self.microwave = builder.build(name="microwave")
        
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'065-b_cups'}",
        )
        builder.inital_pose = sapien.Pose(p=[1, 1, 0])  
        self.cup = builder.build(name="cup") 

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'031_spoon'}",
        )
        builder.inital_pose = sapien.Pose(p=[-1, -1, 0]) 
        self.spoon = builder.build(name="spoon") 

    # def _setup_sensors(self, options: dict):
    #     # default code here will setup all sensors. You can add additional code to change the sensors e.g.
    #     # if you want to randomize camera positions
    #     return super()._setup_sensors()

    # def _load_lighting(self, options: dict):
    #     for scene in self.scene.sub_scenes:
    #         scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
    #         scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
    #         scene.add_directional_light([0, 0, -1], [1, 1, 1])
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.25, -0.2], [-0.2, 0.05]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            # radius = torch.linalg.norm(torch.tensor([0.1, 0.1])) + 0.001
            radius = 0.2
            cup_xy = xy + sampler.sample(radius, 100)
            # min_distance = 0.06
            # spoon_xy = None
            # i = 0
            # while spoon_xy is None or torch.linalg.norm(cup_xy - spoon_xy, dim=1).min() < min_distance:
            #     spoon_xy = xy + sampler.sample(radius, 100, verbose=False)
            #     i += 1
            #     if i >= 100:
            #         cup_xy = xy + sampler.sample(radius, 100, verbose=False)
            #         i = 0
            spoon_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cup_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cup.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = spoon_xy + torch.tensor([0.0, 0.0])
            qs = euler2quat(0, np.pi, 0)
            self.spoon.set_pose(Pose.create_from_pq(p=xyz, q=qs))


            # microwave_region = [[-0.3, 0.4], [0.0, 0.3]]
            # microwave_sampler = randomization.UniformPlacementSampler(
            #     bounds=microwave_region, batch_size=b, device=self.device
            # )
            # microwave_xy = xy + microwave_sampler.sample(radius, 100)
            
            microwave_z = 0.12
            microwave_xy = torch.tensor([-0.2, 0.35])
            xyz[:, :2] = microwave_xy
            xyz[:, 2] = microwave_z
            theta = np.pi / 4
            qs = torch.tensor([np.cos(theta), 0, 0, np.sin(theta)])
            self.microwave.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self.microwave.set_qpos(torch.tensor([0.0, 0.0, 0.0]))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = self.spoon.is_static(lin_thresh=1e-2, ang_thresh=0.5) * self.cup.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_microwave_close = torch.tensor([abs(self.microwave.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.spoon) | self.agent.is_grasping(self.cup)
        pos_spoon = self.spoon.pose.p
        pos_cup = self.cup.pose.p
        pos_microwave = self.microwave.get_pose().p
        offset_spoon_cup = pos_spoon - pos_cup
        is_spoon_in_cup = ( torch.linalg.norm(offset_spoon_cup[..., :2], axis=1) <= 0.05 ) * (torch.abs(offset_spoon_cup[..., 2]) < 0.1)
        offset_cup_microwave = pos_cup - pos_microwave
        is_cup_in_microwave = ( torch.linalg.norm(offset_cup_microwave[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_cup_microwave[..., 2]) < 0.08)
        # print(f'\nis_all_static: {is_all_static}')
        # print(f'is_microwave_close: {is_microwave_close}')
        # print(f'is_something_grasped: {is_something_grasped}')
        # print(f'is_spoon_in_cup: {is_spoon_in_cup}')
        # print(f'is_cup_in_microwave: {is_cup_in_microwave}')
        success = is_all_static * is_microwave_close * (~is_something_grasped) * is_spoon_in_cup * is_cup_in_microwave
        return {
            "is_all_static": is_all_static,
            "is_microwave_close": is_microwave_close,
            "is_something_grasped": is_something_grasped,
            "is_spoon_in_cup": is_spoon_in_cup,
            "is_cup_in_microwave": is_cup_in_microwave,
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

@register_env("MicrowaveTask-fork", max_episode_steps=200)
class MicrowaveTaskEnv_fork(BaseEnv):

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
        # pose = sapien_utils.look_at([0.8, 0., 0.1], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([0.7, -0.4, 0.1], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([0.3, -0.1, 0.8], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([-0.5, -0.5, 0.4], [0.0, 0.0, 0.15])
        pose = sapien_utils.look_at([0.2, 0.2, 0.7], [0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = noTableSceneBuilder_microwave(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.replicaCAD_scene = ReplicaCADSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.replicaCAD_scene.build(1)        
        # 加载微波炉
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{7167}"
        )
        builder.inital_pose = sapien.Pose(p=[10, 10, 10])
        self.microwave = builder.build(name="microwave")
        
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'065-c_cups'}",
        )
        builder.inital_pose = sapien.Pose(p=[1, 1, 0])  
        self.cup = builder.build(name="cup") 

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'030_fork'}",
        )
        builder.inital_pose = sapien.Pose(p=[-1, -1, 0]) 
        self.spoon = builder.build(name="spoon") 
        
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
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.25, -0.2], [-0.2, 0.05]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = 0.2
            cup_xy = xy + sampler.sample(radius, 100)
            spoon_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cup_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cup.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = spoon_xy + torch.tensor([0.0, 0.0])
            qs = euler2quat(0, np.pi, 0)
            self.spoon.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            
            microwave_z = 0.12
            microwave_xy = torch.tensor([-0.2, 0.35])
            xyz[:, :2] = microwave_xy
            xyz[:, 2] = microwave_z
            theta = np.pi / 4
            qs = torch.tensor([np.cos(theta), 0, 0, np.sin(theta)])
            self.microwave.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self.microwave.set_qpos(torch.tensor([0.0, 0.0, 0.0]))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = self.spoon.is_static(lin_thresh=1e-2, ang_thresh=0.5) * self.cup.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_microwave_close = torch.tensor([abs(self.microwave.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.spoon) | self.agent.is_grasping(self.cup)
        pos_spoon = self.spoon.pose.p
        pos_cup = self.cup.pose.p
        pos_microwave = self.microwave.get_pose().p
        offset_spoon_cup = pos_spoon - pos_cup
        is_spoon_in_cup = ( torch.linalg.norm(offset_spoon_cup[..., :2], axis=1) <= 0.05 ) * (torch.abs(offset_spoon_cup[..., 2]) < 0.1)
        offset_cup_microwave = pos_cup - pos_microwave
        is_cup_in_microwave = ( torch.linalg.norm(offset_cup_microwave[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_cup_microwave[..., 2]) < 0.08)
        success = is_all_static * is_microwave_close * (~is_something_grasped) * is_spoon_in_cup * is_cup_in_microwave
        return {
            "is_all_static": is_all_static,
            "is_microwave_close": is_microwave_close,
            "is_something_grasped": is_something_grasped,
            "is_spoon_in_cup": is_spoon_in_cup,
            "is_cup_in_microwave": is_cup_in_microwave,
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

@register_env("MicrowaveTask-mug", max_episode_steps=200)
class MicrowaveTaskEnv_mug(BaseEnv):

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
        # pose = sapien_utils.look_at([0.8, 0., 0.1], [0.0, 0.0, 0.15])
        pose = sapien_utils.look_at([0.7, -0.4, 0.1], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([0.3, -0.1, 0.8], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([-0.5, -0.5, 0.4], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([0.2, 0.2, 0.7], [0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = noTableSceneBuilder_microwave(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([2])      
        # 加载微波炉
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{7167}"
        )
        builder.inital_pose = sapien.Pose(p=[10, 10, 10])
        self.microwave = builder.build(name="microwave")
        
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'025_mug'}",
        )
        builder.inital_pose = sapien.Pose(p=[1, 1, 0], q= euler2quat(0, 0, np.pi/2))  
        self.cup = builder.build(name="cup") 

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'031_spoon'}",
        )
        builder.inital_pose = sapien.Pose(p=[-1, -1, 0]) 
        self.spoon = builder.build(name="spoon") 
        
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
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.25, -0.2], [-0.2, 0.05]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = 0.2
            cup_xy = xy + sampler.sample(radius, 100)
            spoon_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cup_xy
            self.cup.set_pose(Pose.create_from_pq(p=xyz.clone(), q= euler2quat(0, 0, np.pi/2)))

            xyz[:, :2] = spoon_xy + torch.tensor([0.0, 0.0])
            qs = euler2quat(0, np.pi, 0)
            self.spoon.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            
            microwave_z = 0.12
            microwave_xy = torch.tensor([-0.2, 0.35])
            xyz[:, :2] = microwave_xy
            xyz[:, 2] = microwave_z
            theta = np.pi / 4
            qs = torch.tensor([np.cos(theta), 0, 0, np.sin(theta)])
            self.microwave.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self.microwave.set_qpos(torch.tensor([0.0, 0.0, 0.0]))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = self.spoon.is_static(lin_thresh=1e-2, ang_thresh=0.5) * self.cup.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_microwave_close = torch.tensor([abs(self.microwave.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.spoon) | self.agent.is_grasping(self.cup)
        pos_spoon = self.spoon.pose.p
        pos_cup = self.cup.pose.p
        pos_microwave = self.microwave.get_pose().p
        offset_spoon_cup = pos_spoon - pos_cup
        is_spoon_in_cup = ( torch.linalg.norm(offset_spoon_cup[..., :2], axis=1) <= 0.05 ) * (torch.abs(offset_spoon_cup[..., 2]) < 0.1)
        offset_cup_microwave = pos_cup - pos_microwave
        is_cup_in_microwave = ( torch.linalg.norm(offset_cup_microwave[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_cup_microwave[..., 2]) < 0.08)
        success = is_all_static * is_microwave_close * (~is_something_grasped) * is_spoon_in_cup * is_cup_in_microwave
        return {
            "is_all_static": is_all_static,
            "is_microwave_close": is_microwave_close,
            "is_something_grasped": is_something_grasped,
            "is_spoon_in_cup": is_spoon_in_cup,
            "is_cup_in_microwave": is_cup_in_microwave,
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

@register_env("MicrowaveTask-knife", max_episode_steps=200)
class MicrowaveTaskEnv_knife(BaseEnv):

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
        # pose = sapien_utils.look_at([0.8, 0., 0.1], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([0.7, -0.4, 0.1], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([0.3, -0.1, 0.8], [0.0, 0.0, 0.15])
        # pose = sapien_utils.look_at([-0.5, -0.5, 0.4], [0.0, 0.0, 0.15])
        pose = sapien_utils.look_at([0.2, 0.2, 0.7], [0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _load_scene(self, options: dict):
        # 加载桌子
        self.table_scene = noTableSceneBuilder_microwave(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.ai2thor_scene = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build([5])     
        # 加载微波炉
        builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{7167}"
        )
        builder.inital_pose = sapien.Pose(p=[10, 10, 10])
        self.microwave = builder.build(name="microwave")
        
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'065-c_cups'}",
        )
        builder.inital_pose = sapien.Pose(p=[1, 1, 0])  
        self.cup = builder.build(name="cup") 

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{'032_knife'}",
        )
        builder.inital_pose = sapien.Pose(p=[-1, -1, 0]) 
        self.spoon = builder.build(name="spoon") 
        
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
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.25, -0.2], [-0.2, 0.05]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = 0.2
            cup_xy = xy + sampler.sample(radius, 100)
            spoon_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cup_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cup.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = spoon_xy + torch.tensor([0.0, 0.0])
            qs = euler2quat(0, np.pi, 0)
            self.spoon.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            
            microwave_z = 0.12
            microwave_xy = torch.tensor([-0.2, 0.35])
            xyz[:, :2] = microwave_xy
            xyz[:, 2] = microwave_z
            theta = np.pi / 4
            qs = torch.tensor([np.cos(theta), 0, 0, np.sin(theta)])
            self.microwave.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self.microwave.set_qpos(torch.tensor([0.0, 0.0, 0.0]))

    # def evaluate(self, obs: Any):
    def evaluate(self):
        is_all_static = self.spoon.is_static(lin_thresh=1e-2, ang_thresh=0.5) * self.cup.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_microwave_close = torch.tensor([abs(self.microwave.get_qpos()[0, 0]) < 0.01])
        is_something_grasped = self.agent.is_grasping(self.spoon) | self.agent.is_grasping(self.cup)
        pos_spoon = self.spoon.pose.p
        pos_cup = self.cup.pose.p
        pos_microwave = self.microwave.get_pose().p
        offset_spoon_cup = pos_spoon - pos_cup
        is_spoon_in_cup = ( torch.linalg.norm(offset_spoon_cup[..., :2], axis=1) <= 0.05 ) * (torch.abs(offset_spoon_cup[..., 2]) < 0.1)
        offset_cup_microwave = pos_cup - pos_microwave
        is_cup_in_microwave = ( torch.linalg.norm(offset_cup_microwave[..., :2], axis=1) <= 0.1 ) * (torch.abs(offset_cup_microwave[..., 2]) < 0.08)
        success = is_all_static * is_microwave_close * (~is_something_grasped) * is_spoon_in_cup * is_cup_in_microwave
        return {
            "is_all_static": is_all_static,
            "is_microwave_close": is_microwave_close,
            "is_something_grasped": is_something_grasped,
            "is_spoon_in_cup": is_spoon_in_cup,
            "is_cup_in_microwave": is_cup_in_microwave,
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
        