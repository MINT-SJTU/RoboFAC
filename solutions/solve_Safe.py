import argparse
import gymnasium as gym
import numpy as np
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
from tasks.task_Safe import SafeTaskEnv, SafeTaskEnv_usb, SafeTaskEnv_screwdriver, SafeTaskEnv_hammer, SafeTaskEnv_spatula

def solveSafe(env: SafeTaskEnv, seed=None, debug=False, vis=False):
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

    bar_obb = get_actor_obb(env.goldbar)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        bar_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.goldbar.pose.sp.p)
    offset = sapien.Pose([0.04, 0, 0.015])
    grasp_pose = grasp_pose * (offset)

    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()
    # Lift to safe height
    lift_height = 0.2
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    safe_pose = env.safe.get_pose()
    safe_pose.set_q(grasp_pose.q)

    safe_reach_pose = safe_pose * sapien.Pose([0.3, 0.03, 0])
    res = planner.move_to_pose_with_screw(safe_reach_pose)
    if res == -1: return res

    safe_move_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.08])
    res = planner.move_to_pose_with_screw(safe_move_pose)
    if res == -1: return res
    planner.open_gripper()

    leave_pose1 = safe_pose * sapien.Pose([0.2, 0.03, 0.12])
    res = planner.move_to_pose_with_screw(leave_pose1)
    if res == -1: return res

    planner.close_gripper()
    pull_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.1])
    res = planner.move_to_pose_with_screw(pull_pose)
    if res == -1: return res

    
    safe_pose_horizon = safe_pose * sapien.Pose(q=euler2quat(0, - np.pi * 0.5, 0))

    step1_pose = safe_pose_horizon * sapien.Pose([-0.06, 0.05, -0.3])
    step2_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.3])
    step3_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.2])
    close_pose1 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.18])
    close_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.1])
    leave_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.3])
    res = planner.move_to_pose_with_screw(step1_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step2_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step3_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose2)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(leave_pose2)
    if res == -1: return res

    planner.open_gripper()
    safe_pose_vertical = safe_pose_horizon * sapien.Pose(q=euler2quat(0, 0, np.pi * 0.5))

    grasp_pose1 = safe_pose_vertical * sapien.Pose([0.05, 0, -0.13])
    # grasp_pose1 = safe_pose_vertical * sapien.Pose([0.0, 0.05, -0.1])
    reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    # reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(grasp_pose1)
    if res == -1: return res
    planner.close_gripper()

    rotation_pose1 = grasp_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose1)
    if res == -1: return res
    rotation_pose2 = rotation_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose2)
    if res == -1: return res

    # res = planner.open_gripper()

    planner.close()
    # time.sleep(1)
    return res

def solveSafe_usb(env: SafeTaskEnv_usb, seed=None, debug=False, vis=False):
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

    bar_obb = get_actor_obb(env.goldbar)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        bar_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.goldbar.pose.sp.p)
    offset = sapien.Pose([0.08, 0, 0.015])
    grasp_pose = grasp_pose * (offset)

    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05]) 
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()
    # Lift to safe height
    lift_height = 0.2
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    safe_pose = env.safe.get_pose()
    safe_pose.set_q(grasp_pose.q)

    safe_reach_pose = safe_pose * sapien.Pose([0.3, 0.03, 0])
    res = planner.move_to_pose_with_screw(safe_reach_pose)
    if res == -1: return res

    safe_move_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.08])
    res = planner.move_to_pose_with_screw(safe_move_pose)
    if res == -1: return res
    planner.open_gripper()

    leave_pose1 = safe_pose * sapien.Pose([0.2, 0.03, 0.12])
    res = planner.move_to_pose_with_screw(leave_pose1)
    if res == -1: return res

    planner.close_gripper()
    pull_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.1])
    res = planner.move_to_pose_with_screw(pull_pose)
    if res == -1: return res

    safe_pose_horizon = safe_pose * sapien.Pose(q=euler2quat(0, - np.pi * 0.5, 0))

    step1_pose = safe_pose_horizon * sapien.Pose([-0.06, 0.05, -0.3])
    step2_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.3])
    step3_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.2])
    close_pose1 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.18])
    close_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.1])
    leave_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.3])
    res = planner.move_to_pose_with_screw(step1_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step2_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step3_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose2)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(leave_pose2)
    if res == -1: return res

    planner.open_gripper()
    safe_pose_vertical = safe_pose_horizon * sapien.Pose(q=euler2quat(0, 0, np.pi * 0.5))

    grasp_pose1 = safe_pose_vertical * sapien.Pose([0.05, 0, -0.13])
    # grasp_pose1 = safe_pose_vertical * sapien.Pose([0.0, 0.05, -0.1])
    reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    # reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(grasp_pose1)
    if res == -1: return res
    planner.close_gripper()

    rotation_pose1 = grasp_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose1)
    if res == -1: return res
    rotation_pose2 = rotation_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose2)
    if res == -1: return res

    planner.close()
    return res

def solveSafe_screwdriver(env: SafeTaskEnv_screwdriver, seed=None, debug=False, vis=False):
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

    bar_obb = get_actor_obb(env.goldbar)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        bar_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.goldbar.pose.sp.p)
    offset = sapien.Pose([0.07, 0, 0.04]) # 此处为了符合螺丝刀的尺寸修改过
    grasp_pose = grasp_pose * (offset)

    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.06])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()
    # Lift to safe height
    lift_height = 0.2
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    safe_pose = env.safe.get_pose()
    safe_pose.set_q(grasp_pose.q)
    safe_reach_pose = safe_pose * sapien.Pose([0.3, 0.03, 0])
    res = planner.move_to_pose_with_screw(safe_reach_pose)
    if res == -1: return res
    safe_move_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.08])
    res = planner.move_to_pose_with_screw(safe_move_pose)
    if res == -1: return res
    planner.open_gripper()
    leave_pose1 = safe_pose * sapien.Pose([0.2, 0.03, 0.12])
    res = planner.move_to_pose_with_screw(leave_pose1)
    if res == -1: return res
    planner.close_gripper()
    pull_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.1])
    res = planner.move_to_pose_with_screw(pull_pose)
    if res == -1: return res

    safe_pose_horizon = safe_pose * sapien.Pose(q=euler2quat(0, - np.pi * 0.5, 0))

    step1_pose = safe_pose_horizon * sapien.Pose([-0.06, 0.05, -0.3])
    step2_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.3])
    step3_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.2])
    close_pose1 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.18])
    close_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.1])
    leave_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.3])
    res = planner.move_to_pose_with_screw(step1_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step2_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step3_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose2)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(leave_pose2)
    if res == -1: return res

    planner.open_gripper()
    safe_pose_vertical = safe_pose_horizon * sapien.Pose(q=euler2quat(0, 0, np.pi * 0.5))

    grasp_pose1 = safe_pose_vertical * sapien.Pose([0.05, 0, -0.13])
    # grasp_pose1 = safe_pose_vertical * sapien.Pose([0.0, 0.05, -0.1])
    reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    # reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(grasp_pose1)
    if res == -1: return res
    planner.close_gripper()

    rotation_pose1 = grasp_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose1)
    if res == -1: return res
    rotation_pose2 = rotation_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose2)
    if res == -1: return res

    planner.close()
    return res

def solveSafe_hammer(env: SafeTaskEnv_hammer, seed=None, debug=False, vis=False):
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

    bar_obb = get_actor_obb(env.goldbar)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        bar_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.goldbar.pose.sp.p)
    offset = sapien.Pose([0.07, 0, 0.04]) # 此处为了符合锤子的尺寸修改过
    grasp_pose = grasp_pose * (offset)


    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.06])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()
    # Lift to safe height
    lift_height = 0.2
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    
    safe_pose = env.safe.get_pose()
    safe_pose.set_q(grasp_pose.q)
    
    safe_reach_pose = safe_pose * sapien.Pose([0.3, 0.03, 0])
    res = planner.move_to_pose_with_screw(safe_reach_pose)
    if res == -1: return res
    
    safe_move_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.08])
    res = planner.move_to_pose_with_screw(safe_move_pose)
    if res == -1: return res
    planner.open_gripper()
    
    leave_pose1 = safe_pose * sapien.Pose([0.2, 0.03, 0.12])
    res = planner.move_to_pose_with_screw(leave_pose1)
    if res == -1: return res
    
    planner.close_gripper()
    pull_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.1])
    res = planner.move_to_pose_with_screw(pull_pose)
    if res == -1: return res

    
    safe_pose_horizon = safe_pose * sapien.Pose(q=euler2quat(0, - np.pi * 0.5, 0))

    step1_pose = safe_pose_horizon * sapien.Pose([-0.06, 0.05, -0.3])
    step2_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.3])
    step3_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.2])
    close_pose1 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.18])
    close_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.1])
    leave_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.3])
    res = planner.move_to_pose_with_screw(step1_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step2_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step3_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose2)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(leave_pose2)
    if res == -1: return res

    
    planner.open_gripper()
    safe_pose_vertical = safe_pose_horizon * sapien.Pose(q=euler2quat(0, 0, np.pi * 0.5))

    grasp_pose1 = safe_pose_vertical * sapien.Pose([0.05, 0, -0.13])
    # grasp_pose1 = safe_pose_vertical * sapien.Pose([0.0, 0.05, -0.1])
    reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    # reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(grasp_pose1)
    if res == -1: return res
    planner.close_gripper()

    rotation_pose1 = grasp_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose1)
    if res == -1: return res
    rotation_pose2 = rotation_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose2)
    if res == -1: return res

    planner.close()
    return res

def solveSafe_spatula(env: SafeTaskEnv_spatula, seed=None, debug=False, vis=False):
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

    bar_obb = get_actor_obb(env.goldbar)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        bar_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.goldbar.pose.sp.p)
    offset = sapien.Pose([0.07, 0, 0.04]) # 此处为了符合锤子的尺寸修改过
    grasp_pose = grasp_pose * (offset)


    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.06])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()
    # Lift to safe height
    lift_height = 0.2
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    
    safe_pose = env.safe.get_pose()
    safe_pose.set_q(grasp_pose.q)
    
    safe_reach_pose = safe_pose * sapien.Pose([0.3, 0.03, 0])
    res = planner.move_to_pose_with_screw(safe_reach_pose)
    if res == -1: return res
    
    safe_move_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.08])
    res = planner.move_to_pose_with_screw(safe_move_pose)
    if res == -1: return res
    planner.open_gripper()
    
    leave_pose1 = safe_pose * sapien.Pose([0.2, 0.03, 0.12])
    res = planner.move_to_pose_with_screw(leave_pose1)
    if res == -1: return res
    
    planner.close_gripper()
    pull_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.12])
    res = planner.move_to_pose_with_screw(pull_pose)
    if res == -1: return res

    
    safe_pose_horizon = safe_pose * sapien.Pose(q=euler2quat(0, - np.pi * 0.5, 0))

    step1_pose = safe_pose_horizon * sapien.Pose([-0.06, 0.05, -0.3])
    step2_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.3])
    step3_pose = safe_pose_horizon * sapien.Pose([-0.06, -0.28, -0.2])
    close_pose1 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.18])
    close_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.1])
    leave_pose2 = safe_pose_horizon * sapien.Pose([-0.06, 0.0, -0.3])
    res = planner.move_to_pose_with_screw(step1_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step2_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(step3_pose)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(close_pose2)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(leave_pose2)
    if res == -1: return res

    
    planner.open_gripper()
    safe_pose_vertical = safe_pose_horizon * sapien.Pose(q=euler2quat(0, 0, np.pi * 0.5))

    grasp_pose1 = safe_pose_vertical * sapien.Pose([0.05, 0, -0.13])
    # grasp_pose1 = safe_pose_vertical * sapien.Pose([0.0, 0.05, -0.1])
    reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    # reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose1)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(grasp_pose1)
    if res == -1: return res
    planner.close_gripper()

    rotation_pose1 = grasp_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose1)
    if res == -1: return res
    rotation_pose2 = rotation_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    res = planner.move_to_pose_with_screw(rotation_pose2)
    if res == -1: return res

    planner.close()
    return res

if __name__ == "__main__":
    env = gym.make(
        "SafeTask-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(10):
        solveSafe(env, seed=seed, debug=False, vis=True)