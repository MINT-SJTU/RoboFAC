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
from tasks.task_UprightStack import UprightStackEnv

def solveUprightStack(env: UprightStackEnv, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.brick)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.06, 0, 0]) 
    grasp_pose = grasp_pose * offset
    grasp_angle = np.deg2rad(0)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1]) 
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    # planner.close_gripper(gripper_state=-0.2)
    planner.close_gripper(gripper_state=-0.28)
    # Lift
    lift_pose = sapien.Pose([0.0, 0, 0.3]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    time.sleep(0.1)
    if res == -1: return res
    # Move
    target_pose = env.cubeA.pose * sapien.Pose([0.02, 0, 0.25])
    target_pose.q = lift_pose.q
    res = planner.move_to_pose_with_screw(target_pose)
    time.sleep(0.1)
    if res == -1: return res
    # Place upright
    # theta = np.pi * 0.086
    theta = np.pi * 0.108
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = target_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res
    time.sleep(0.1)
    # Lower
    lower_pose =  env.cubeA.pose * sapien.Pose([0.02, 0, 0.155])
    lower_pose.q = final_pose.q
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res
    time.sleep(0.1)
    planner.open_gripper()
    time.sleep(1)

    planner.close()
    return res
def solveUprightStack_gen1(env, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.brick)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.06, 0, 0])
    grasp_pose = grasp_pose * offset
    grasp_angle = np.deg2rad(0)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    # planner.close_gripper(gripper_state=-0.2)
    planner.close_gripper(gripper_state=-0.28)
    # Lift
    lift_pose = sapien.Pose([0.0, 0, 0.3]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    time.sleep(0.5)
    if res == -1: return res
    # Move
    target_pose = env.cubeA.pose * sapien.Pose([0.02, 0, 0.25])
    target_pose.q = lift_pose.q
    res = planner.move_to_pose_with_screw(target_pose)
    time.sleep(0.5)
    if res == -1: return res
    # Place upright
    # theta = np.pi * 0.086
    theta = np.pi * 0.108
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = target_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res
    time.sleep(0.5)
    # Lower
    lower_pose =  env.cubeA.pose * sapien.Pose([0.02, 0, 0.155])
    lower_pose.q = final_pose.q
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res
    time.sleep(1)

    planner.close()
    return res

def solveUprightStack_gen2(env, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.brick)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.06, 0, 0])
    grasp_pose = grasp_pose * offset
    grasp_angle = np.deg2rad(0)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    # planner.close_gripper(gripper_state=-0.2)
    planner.close_gripper(gripper_state=-0.28)
    # Lift
    lift_pose = sapien.Pose([0.0, 0, 0.3]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    time.sleep(0.5)
    if res == -1: return res
    # Move
    target_pose = env.cubeA.pose * sapien.Pose([0.02, 0, 0.25])
    target_pose.q = lift_pose.q
    res = planner.move_to_pose_with_screw(target_pose)
    time.sleep(0.5)
    if res == -1: return res
    # Place upright
    # theta = np.pi * 0.086
    theta = np.pi * 0.108
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = target_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res
    time.sleep(0.5)
    # Lower
    lower_pose =  env.cubeA.pose * sapien.Pose([0.02, 0, 0.155])
    lower_pose.q = final_pose.q
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res
    time.sleep(1)

    planner.close()
    return res
if __name__ == "__main__":
    env = gym.make(
        "UprightStack-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(20):
        # solveTools(env, seed=seed, debug=True, vis=True)
        # res = solve_with_errors(env,log_file='test',error_stage="lift_tool",perturbation_type="speed_offset", vis=True)
        res = solveUprightStack(env,vis=True)
        time.sleep(1)