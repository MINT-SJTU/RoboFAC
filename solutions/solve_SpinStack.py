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
from tasks.task_SpinStack import *
from mani_skill.utils.structs.pose import Pose
def main(): 
    env = gym.make(
        "SpinStack-v1",
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(10):
        solveRotation(env, seed=seed, debug=False, vis=True)

def solveRotation(env, seed=None, debug=False, vis=False):
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

    for i in range(10):
        planner.skip_step()
    obb = get_actor_obb(env.cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    grasp_pose.p = torch.tensor([env.disk_x - env.cube_r, env.disk_y, env.cube.pose.sp.p[2]])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cube.pose.sp.p[0] - env.disk_x) / (env.cube.pose.sp.p[1] - env.disk_y))
        if (env.cube.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle
        if angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            res = planner.move_to_pose_with_screw(grasp_pose)
            planner.close_gripper()
            break
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose)

    goal_pose = Pose.create_from_pq(p=torch.tensor([-0.1, -0.1, 0.08]), q=lift_pose.q)
    offset = (goal_pose.p - env.cube.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    lift_pose2 = align_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose2)

    grasp_pose.p = torch.tensor([env.disk_x - env.cubeB_r, env.disk_y, env.cubeB.pose.sp.p[2]])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cubeB.pose.sp.p[0] - env.disk_x) / (env.cubeB.pose.sp.p[1] - env.disk_y))
        if (env.cubeB.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle
        if angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            res = planner.move_to_pose_with_screw(grasp_pose)
            if res == -1:
                return res
            planner.close_gripper()
            break

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose)
    # goal_pose = env.cube.pose * sapien.Pose([0, 0, 0.05]) # 放置在cube上
    offset = (goal_pose.p - env.cubeB.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()   
    
    for i in range(10):
        planner.skip_step()

    planner.close()
    return res
def solveRotation_gen1(env, seed=None, debug=False, vis=False):
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

    for i in range(10):
        planner.skip_step()

    obb = get_actor_obb(env.cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    grasp_pose.p = torch.tensor([env.disk_x - env.cube_r, env.disk_y, env.cube.pose.sp.p[2]])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])

    planner.move_to_pose_with_screw(reach_pose)

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cube.pose.sp.p[0] - env.disk_x) / (env.cube.pose.sp.p[1] - env.disk_y))
        if (env.cube.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle
        if angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            res = planner.move_to_pose_with_screw(grasp_pose)
            planner.close_gripper()
            break
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose)

    goal_pose = Pose.create_from_pq(p=torch.tensor([-0.1, -0.1, 0.08]), q=lift_pose.q)
    offset = (goal_pose.p - env.cube.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    lift_pose2 = align_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose2)

    grasp_pose.p = torch.tensor([env.disk_x - env.cubeB_r, env.disk_y, env.cubeB.pose.sp.p[2]])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cubeB.pose.sp.p[0] - env.disk_x) / (env.cubeB.pose.sp.p[1] - env.disk_y))
        if (env.cubeB.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle
        if angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            res = planner.move_to_pose_with_screw(grasp_pose)
            if res == -1:
                return res
            planner.close_gripper()
            break

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.cube.pose * sapien.Pose([0, 0, 0.08]) 
    offset = (goal_pose.p - env.cubeB.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()   
    
    for i in range(10):
        planner.skip_step()

    planner.close()
    return res

def solveRotation_gen2(env, seed=None, debug=False, vis=False):
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

    for i in range(10):
        planner.skip_step()
    obb = get_actor_obb(env.cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    grasp_pose.p = torch.tensor([env.disk_x - env.cube_r, env.disk_y, env.cube.pose.sp.p[2]])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])

    planner.move_to_pose_with_screw(reach_pose)

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cube.pose.sp.p[0] - env.disk_x) / (env.cube.pose.sp.p[1] - env.disk_y))
        if (env.cube.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle
        if angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            res = planner.move_to_pose_with_screw(grasp_pose)
            planner.close_gripper()
            break
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose)

    goal_pose = Pose.create_from_pq(p=torch.tensor([-0.1, -0.1, 0.08]), q=lift_pose.q)
    offset = (goal_pose.p - env.cube.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    lift_pose2 = align_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose2)

    grasp_pose.p = torch.tensor([env.disk_x - env.cubeB_r, env.disk_y, env.cubeB.pose.sp.p[2]])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cubeB.pose.sp.p[0] - env.disk_x) / (env.cubeB.pose.sp.p[1] - env.disk_y))
        if (env.cubeB.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle
        if angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            res = planner.move_to_pose_with_screw(grasp_pose)
            if res == -1:
                return res
            planner.close_gripper()
            break

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.cube.pose * sapien.Pose([0, 0, 0.08]) 
    offset = (goal_pose.p - env.cubeB.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()   
    
    for i in range(10):
        planner.skip_step()

    planner.close()
    return res
if __name__ == "__main__":
    main()