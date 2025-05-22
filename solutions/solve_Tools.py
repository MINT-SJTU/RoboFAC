import argparse
import gymnasium as gym
import numpy as np
import trimesh
import sapien
from transforms3d.euler import euler2quat
import torch # Add 
import time

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.task_Tools import ToolsTaskEnv

def solveTools(env: ToolsTaskEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # Get tool OBB and compute grasp pose
    tool_obb = get_actor_obb(env.circular_shape_tool)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        tool_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.circular_shape_tool.pose.sp.p)
    offset = sapien.Pose([0.02, 0, 0])
    grasp_pose = grasp_pose * (offset)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift tool to safe height
    # -------------------------------------------------------------------------- #
    lift_height = 0.35  
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    cube_pos = env.charger.pose.sp.p
    approach_offset = sapien.Pose(
        [-(env.handle_length + env.cube_half_size - 0.14),  
        -0.0,  
        lift_height - 0.05]  
    )
    approach_pose = sapien.Pose(cube_pos) * approach_offset
    approach_pose.set_q(grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(approach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower tool behind cube
    # -------------------------------------------------------------------------- #
    behind_offset = sapien.Pose(
        [-(env.handle_length + env.cube_half_size - 0.14),  
        -0.035,  
        0.02] 
    )
    hook_pose = sapien.Pose(cube_pos) * behind_offset
    hook_pose.set_q(grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(hook_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Pull cube
    # -------------------------------------------------------------------------- #
    pull_offset = sapien.Pose([-0.35, 0, 0])
    target_pose = hook_pose * pull_offset
    res = planner.move_to_pose_with_screw(target_pose)
    if res == -1: return res

    push_pose = target_pose * sapien.Pose([0.1, 0.1, 0])
    res = planner.move_to_pose_with_screw(push_pose)
    if res == -1: return res

    planner.open_gripper()

    lift_pose = push_pose * sapien.Pose([0, 0, -0.2])
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    charger_base_pose = env.charger_base_pose
    charger_base_size = np.array(env.unwrapped._base_size) * 2

    obb = trimesh.primitives.Box(
        extents=charger_base_size,
        transform=charger_base_pose.sp.to_transformation_matrix(),
    )

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # add a angle to grasp
    grasp_angle = np.deg2rad(15)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Align
    # -------------------------------------------------------------------------- #
    pre_insert_pose = (
        env.goal_pose.sp
        * sapien.Pose([-0.05, 0.0, 0.0])
        * env.charger.pose.sp.inv()
        * env.agent.tcp.pose.sp
    )
    insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp * sapien.Pose([0.0, 0.0, 0.0])
    planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0)
    planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=10)
    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(insert_pose)


    planner.close()
    return res
