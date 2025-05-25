import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import time

from mani_skill.envs.tasks import LiftPegUprightEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.generalization.generalization.task_generalization.task_LiftPegUpright import LiftPegUprightEnv_box, LiftPegUprightEnv_can, LiftPegUprightEnv_cup

from tasks.task_LiftPegUpright import *

def solve_liftpegupright_box(env: LiftPegUprightEnv_box, seed=None, debug=False, vis=False):
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = env.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.05, 0, -0.05]) 
    grasp_pose = grasp_pose * offset
    grasp_angle = np.deg2rad(0)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1]) 
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=-0.6)

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0.0, 0, 0.3]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Place upright
    # -------------------------------------------------------------------------- #
    theta = np.pi * 0.1
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    
    final_pose = lift_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    lower_pose = sapien.Pose([0, 0, -0.1]) * final_pose
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res

    planner.close()
    
    planner.open_gripper()
    time.sleep(0.5)
    return res


def solve_liftpegupright_can(env: LiftPegUprightEnv_can, seed=None, debug=False, vis=False):
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = env.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.03, 0, 0.02]) 
    grasp_pose = grasp_pose * offset
    grasp_angle = np.deg2rad(0)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=0.36) 

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([-0.1, 0, 0.2]) * grasp_pose 
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Place upright
    # -------------------------------------------------------------------------- #
    theta = np.pi * 0.05 
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    
    final_pose = lift_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    lower_pose = sapien.Pose([0, 0, -0.10]) * final_pose
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res

    planner.close()
    
    planner.open_gripper()
    return res
def solve_liftpegupright_can(env: LiftPegUprightEnv_can, seed=None, debug=False, vis=False):
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = env.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.03, 0, 0.02]) 
    grasp_pose = grasp_pose * offset
    grasp_angle = np.deg2rad(0)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=0.36) 

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([-0.1, 0, 0.2]) * grasp_pose 
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Place upright
    # -------------------------------------------------------------------------- #
    theta = np.pi * 0.05 
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    
    final_pose = lift_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    lower_pose = sapien.Pose([0, 0, -0.10]) * final_pose
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res

    planner.close()
    
    planner.open_gripper()
    return res
    
def solve_liftpegupright_cup(env: LiftPegUprightEnv_cup, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    record = {}
    record['stage'] = ''
    # record['mode'] = mode
    record['error_type'] = 'error_type'
    record['success_level'] = ''
    record['detail'] = 'desc'
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = env.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.02, 0, 0.02]) 
    grasp_pose = grasp_pose * offset
    grasp_angle = np.deg2rad(0)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=0.63) 

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose 
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Place upright
    # -------------------------------------------------------------------------- #
    theta = np.pi * 0.05 
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    
    final_pose = lift_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    lower_pose = sapien.Pose([0, 0, -0.10]) * final_pose
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res

    planner.close()
    
    planner.open_gripper()
    time.sleep(0.5)
    # return res, record # ????????????????
    return res