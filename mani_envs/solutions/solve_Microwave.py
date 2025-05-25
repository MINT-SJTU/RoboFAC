import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import torch # Add 
import time
import sys
import os

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.task_Microwave import MicrowaveTaskEnv, MicrowaveTaskEnv_fork, MicrowaveTaskEnv_mug, MicrowaveTaskEnv_knife
from tasks import *
def solveMicrowave(env: MicrowaveTaskEnv, seed=None, debug=False, vis=False):
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

    # ---------------抓取并竖直spoon------------
    obb = get_actor_obb(env.spoon)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    spoon_init_pose = env.spoon.pose
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.025, 0.0, 0.04])
    grasp_pose = grasp_pose * offset
    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper(gripper_state=-0.6)
    # Lift
    lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    # 靠近一点方便竖直
    lift_pose2 = lift_pose * sapien.Pose([0.18, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose2)
    # upright
    theta = np.pi * 0.225
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = lift_pose2 * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    planner.move_to_pose_with_screw(final_pose)
    # Move to cup
    goal_pose = env.cup.pose * sapien.Pose([0, 0, 0.12])
    # offset = (goal_pose.p - env.spoon.pose.p).numpy()[0]
    # align_pose = sapien.Pose(lift_pose.p + offset, final_pose.q)
    align_pose = sapien.Pose(goal_pose.p.numpy()[0], final_pose.q)
    planner.move_to_pose_with_RRTConnect(align_pose)
    align_pose2 = align_pose * sapien.Pose([-0.048, 0.0, 0.0])
    planner.move_to_pose_with_RRTConnect(align_pose2)
    planner.open_gripper()  
    # -----------------------------------------------  

    # ---------------挪动cup，防止挡门---------------------
    # 抬起一点
    lift_pose2 = align_pose2 * sapien.Pose([0.1, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)
    # 抓取cup
    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    grasp_pose.p[2] = 0.05

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.03])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([0.03, 0.0, 0.0])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.5, -0.18, -0.1])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    # -----------------------------------------------

    # ---------------把microwave打开-----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    # pull the door of the microwave
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.1, -0.152, 0.18])
    offset = (goal_pose.p - align_pose2.p).numpy()[0]
    pull_pose_up = sapien.Pose(grasp_pose.p + offset, grasp_pose.q)
    planner.move_to_pose_with_screw(pull_pose_up)
    pull_pose = pull_pose_up * sapien.Pose([-0.05, 0.0, 0.0])
    planner.move_to_pose_with_screw(pull_pose)
    pull_pose2 = pull_pose * sapien.Pose([0.0, -0.16, -0.1])
    planner.move_to_pose_with_screw(pull_pose2)
    pull_pose3 = pull_pose2 * sapien.Pose([0.0, -0.1, -0.16])
    planner.move_to_pose_with_screw(pull_pose3)

    # -----------杯子放到microwave中----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([-0.05, 0.0, 0.05])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.25, -0.15, -0.05])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    move_pose = align_pose * sapien.Pose([0.02, -0.16, 0.0])
    planner.move_to_pose_with_screw(move_pose)
    move_pose2 = move_pose * sapien.Pose([0.0, 0.0, 0.18])
    planner.move_to_pose_with_screw(move_pose2)
    planner.open_gripper()
    # ----------------------------------------------

    # ---------------把microwave关上-----------------
    # 爪子移到外侧
    depart_pose = move_pose2 * sapien.Pose([0, 0, -0.4])
    planner.move_to_pose_with_screw(depart_pose)
    depart_pose2 = depart_pose * sapien.Pose([0, -0.2, 0])
    planner.move_to_pose_with_screw(depart_pose2)
    depart_pose3 = depart_pose2 * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(depart_pose3)
    # 关门
    push_pose = depart_pose3 * sapien.Pose([0, 0.2, 0])
    planner.move_to_pose_with_screw(push_pose)
    push_pose2 = push_pose * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(push_pose2)
    # --------------------------------------------------

    res = planner.close_gripper()
    planner.close()
    # sleep(1)
    return res

def solveMicrowave_fork(env: MicrowaveTaskEnv_fork, seed=None, debug=False, vis=False):
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

    # ---------------抓取并竖直spoon------------
    obb = get_actor_obb(env.spoon)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    spoon_init_pose = env.spoon.pose
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.025, 0.0, 0.04])
    grasp_pose = grasp_pose * offset
    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper(gripper_state=-0.6)
    # Lift
    lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    # 靠近一点方便竖直
    lift_pose2 = lift_pose * sapien.Pose([0.18, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose2)
    # upright
    theta = np.pi * 0.225
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = lift_pose2 * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    planner.move_to_pose_with_screw(final_pose)
    # Move to cup
    goal_pose = env.cup.pose * sapien.Pose([0, 0, 0.12])
    # offset = (goal_pose.p - env.spoon.pose.p).numpy()[0]
    # align_pose = sapien.Pose(lift_pose.p + offset, final_pose.q)
    align_pose = sapien.Pose(goal_pose.p.numpy()[0], final_pose.q)
    planner.move_to_pose_with_RRTConnect(align_pose)
    align_pose2 = align_pose * sapien.Pose([-0.048, 0.0, 0.0])
    planner.move_to_pose_with_RRTConnect(align_pose2)
    planner.open_gripper()  
    # -----------------------------------------------  

    # ---------------挪动cup，防止挡门---------------------
    # 抬起一点
    lift_pose2 = align_pose2 * sapien.Pose([0.1, 0, 0])
    planner.move_to_pose_with_screw(lift_pose2)
    # 抓取cup
    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    grasp_pose.p[2] = 0.05
    grasp_pose.q = [-0.5, 0.5, 0.5, 0.5]

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.03])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([0.03, 0.0, 0.0])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.5, -0.18, -0.1])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    # -----------------------------------------------

    # ---------------把microwave打开-----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    # pull the door of the microwave
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.1, -0.152, 0.18])
    offset = (goal_pose.p - align_pose2.p).numpy()[0]
    pull_pose_up = sapien.Pose(grasp_pose.p + offset, grasp_pose.q)
    planner.move_to_pose_with_screw(pull_pose_up)
    pull_pose = pull_pose_up * sapien.Pose([-0.05, 0.0, 0.0])
    planner.move_to_pose_with_screw(pull_pose)
    pull_pose2 = pull_pose * sapien.Pose([0.0, -0.16, -0.1])
    planner.move_to_pose_with_screw(pull_pose2)
    pull_pose3 = pull_pose2 * sapien.Pose([0.0, -0.1, -0.16])
    planner.move_to_pose_with_screw(pull_pose3)

    # -----------杯子放到microwave中----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    # grasp_pose.p[2] = 0.05
    # grasp_pose.q = [-0.5, 0.5, 0.5, 0.5]

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([-0.05, 0.0, 0.05])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.25, -0.15, -0.05])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    move_pose = align_pose * sapien.Pose([0.02, -0.16, 0.0])
    planner.move_to_pose_with_screw(move_pose)
    move_pose2 = move_pose * sapien.Pose([0.0, 0.0, 0.18])
    planner.move_to_pose_with_screw(move_pose2)
    planner.open_gripper()
    # ----------------------------------------------

    # ---------------把microwave关上-----------------
    # 爪子移到外侧
    depart_pose = move_pose2 * sapien.Pose([0, 0, -0.4])
    planner.move_to_pose_with_screw(depart_pose)
    depart_pose2 = depart_pose * sapien.Pose([0, -0.2, 0])
    planner.move_to_pose_with_screw(depart_pose2)
    depart_pose3 = depart_pose2 * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(depart_pose3)
    # 关门
    push_pose = depart_pose3 * sapien.Pose([0, 0.2, 0])
    planner.move_to_pose_with_screw(push_pose)
    push_pose2 = push_pose * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(push_pose2)
    # --------------------------------------------------

    res = planner.close_gripper()
    planner.close()
    return res

def solveMicrowave_mug(env: MicrowaveTaskEnv_mug, seed=None, debug=False, vis=False):
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

    # ---------------抓取并竖直spoon------------
    obb = get_actor_obb(env.spoon)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    spoon_init_pose = env.spoon.pose
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.025, 0.0, 0.04])
    grasp_pose = grasp_pose * offset
    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper(gripper_state=-0.6)
    # Lift
    lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    # 靠近一点方便竖直
    lift_pose2 = lift_pose * sapien.Pose([0.18, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose2)
    # upright
    theta = np.pi * 0.225
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = lift_pose2 * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    planner.move_to_pose_with_screw(final_pose)
    # Move to cup
    goal_pose = env.cup.pose * sapien.Pose([0, 0, 0.12])
    # offset = (goal_pose.p - env.spoon.pose.p).numpy()[0]
    # align_pose = sapien.Pose(lift_pose.p + offset, final_pose.q)
    align_pose = sapien.Pose(goal_pose.p.numpy()[0], final_pose.q)
    planner.move_to_pose_with_RRTConnect(align_pose)
    align_pose2 = align_pose * sapien.Pose([-0.048, 0.0, 0.0])
    planner.move_to_pose_with_RRTConnect(align_pose2)
    planner.open_gripper()  
    # -----------------------------------------------  

    # ---------------挪动cup，防止挡门---------------------
    # 抬起一点
    lift_pose2 = align_pose2 * sapien.Pose([0.1, 0, 0])
    planner.move_to_pose_with_screw(lift_pose2)
    # 抓取cup
    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    grasp_pose.p[2] = 0.05
    grasp_pose.q = [-0.5, 0.5, 0.5, 0.5]

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.03])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([0.03, 0.0, 0.0])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.5, -0.18, -0.1])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    # -----------------------------------------------

    # ---------------把microwave打开-----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    # pull the door of the microwave
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.1, -0.152, 0.18])
    offset = (goal_pose.p - align_pose2.p).numpy()[0]
    pull_pose_up = sapien.Pose(grasp_pose.p + offset, grasp_pose.q)
    planner.move_to_pose_with_screw(pull_pose_up)
    pull_pose = pull_pose_up * sapien.Pose([-0.05, 0.0, 0.0])
    planner.move_to_pose_with_screw(pull_pose)
    pull_pose2 = pull_pose * sapien.Pose([0.0, -0.16, -0.1])
    planner.move_to_pose_with_screw(pull_pose2)
    pull_pose3 = pull_pose2 * sapien.Pose([0.0, -0.1, -0.16])
    planner.move_to_pose_with_screw(pull_pose3)

    # -----------杯子放到microwave中----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    grasp_pose.p[2] = 0.05
    grasp_pose.q = [-0.5, 0.5, 0.5, 0.5]

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([-0.05, 0.0, 0.05])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.25, -0.15, -0.05])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    move_pose = align_pose * sapien.Pose([0.02, -0.16, 0.0])
    planner.move_to_pose_with_screw(move_pose)
    move_pose2 = move_pose * sapien.Pose([0.0, 0.0, 0.18])
    planner.move_to_pose_with_screw(move_pose2)
    planner.open_gripper()
    # ----------------------------------------------

    # ---------------把microwave关上-----------------
    # 爪子移到外侧
    depart_pose = move_pose2 * sapien.Pose([0, 0, -0.4])
    planner.move_to_pose_with_screw(depart_pose)
    depart_pose2 = depart_pose * sapien.Pose([0, -0.2, 0])
    planner.move_to_pose_with_screw(depart_pose2)
    depart_pose3 = depart_pose2 * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(depart_pose3)
    # 关门
    push_pose = depart_pose3 * sapien.Pose([0, 0.2, 0])
    planner.move_to_pose_with_screw(push_pose)
    push_pose2 = push_pose * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(push_pose2)
    # --------------------------------------------------

    res = planner.close_gripper()
    planner.close()
    return res

def solveMicrowave_knife(env: MicrowaveTaskEnv_knife, seed=None, debug=False, vis=False):
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

    # ---------------抓取并竖直spoon------------
    obb = get_actor_obb(env.spoon)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    spoon_init_pose = env.spoon.pose
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.025, 0.0, 0.04])
    grasp_pose = grasp_pose * offset
    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper(gripper_state=-0.6)
    # Lift
    lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    # 靠近一点方便竖直
    lift_pose2 = lift_pose * sapien.Pose([0.18, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose2)
    # upright
    theta = np.pi * 0.225
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = lift_pose2 * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    planner.move_to_pose_with_screw(final_pose)
    # Move to cup
    goal_pose = env.cup.pose * sapien.Pose([0, 0, 0.12])
    # offset = (goal_pose.p - env.spoon.pose.p).numpy()[0]
    # align_pose = sapien.Pose(lift_pose.p + offset, final_pose.q)
    align_pose = sapien.Pose(goal_pose.p.numpy()[0], final_pose.q)
    planner.move_to_pose_with_RRTConnect(align_pose)
    align_pose2 = align_pose * sapien.Pose([-0.048, 0.0, 0.0])
    planner.move_to_pose_with_RRTConnect(align_pose2)
    planner.open_gripper()  
    # -----------------------------------------------  

    # ---------------挪动cup，防止挡门---------------------
    # 抬起一点
    lift_pose2 = align_pose2 * sapien.Pose([0.1, 0, 0])
    planner.move_to_pose_with_screw(lift_pose2)
    # 抓取cup
    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    grasp_pose.p[2] = 0.05
    grasp_pose.q = [-0.5, 0.5, 0.5, 0.5]

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.03])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([0.03, 0.0, 0.0])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.5, -0.18, -0.1])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    # -----------------------------------------------

    # ---------------把microwave打开-----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    # pull the door of the microwave
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.1, -0.152, 0.18])
    offset = (goal_pose.p - align_pose2.p).numpy()[0]
    pull_pose_up = sapien.Pose(grasp_pose.p + offset, grasp_pose.q)
    planner.move_to_pose_with_screw(pull_pose_up)
    pull_pose = pull_pose_up * sapien.Pose([-0.05, 0.0, 0.0])
    planner.move_to_pose_with_screw(pull_pose)
    pull_pose2 = pull_pose * sapien.Pose([0.0, -0.16, -0.1])
    planner.move_to_pose_with_screw(pull_pose2)
    pull_pose3 = pull_pose2 * sapien.Pose([0.0, -0.1, -0.16])
    planner.move_to_pose_with_screw(pull_pose3)

    # -----------杯子放到microwave中----------------
    # 抬起一点
    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    obb = get_actor_obb(env.cup)
    approaching = np.array([0, 1, 0]) 
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # 抓取cup
    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([-0.05, 0.0, 0.05])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    # move cup
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.25, -0.15, -0.05])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    move_pose = align_pose * sapien.Pose([0.02, -0.16, 0.0])
    planner.move_to_pose_with_screw(move_pose)
    move_pose2 = move_pose * sapien.Pose([0.0, 0.0, 0.18])
    planner.move_to_pose_with_screw(move_pose2)
    planner.open_gripper()
    # ----------------------------------------------

    # ---------------把microwave关上-----------------
    # 爪子移到外侧
    depart_pose = move_pose2 * sapien.Pose([0, 0, -0.4])
    planner.move_to_pose_with_screw(depart_pose)
    depart_pose2 = depart_pose * sapien.Pose([0, -0.2, 0])
    planner.move_to_pose_with_screw(depart_pose2)
    depart_pose3 = depart_pose2 * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(depart_pose3)
    # 关门
    push_pose = depart_pose3 * sapien.Pose([0, 0.2, 0])
    planner.move_to_pose_with_screw(push_pose)
    push_pose2 = push_pose * sapien.Pose([0, 0, 0.2])
    planner.move_to_pose_with_screw(push_pose2)
    # --------------------------------------------------

    res = planner.close_gripper()
    planner.close()
    return res


if __name__ == "__main__":
    env = gym.make(
        "MicrowaveTask", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(10):
        solveMicrowave(env, seed=seed, debug=False, vis=True)