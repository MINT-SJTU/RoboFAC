import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.task_PushCube import PushCubeEnv_box, PushCubeEnv_cup, PushCubeEnv_toy


def solve_pushcube_box(env: PushCubeEnv_box, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.1, 0, 0]), q=env.agent.tcp.pose.sp.q) 
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res

def solve_pushcube_cup(env: PushCubeEnv_cup, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.08, 0, 0]), q=env.agent.tcp.pose.sp.q) 
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res

def solve_pushcube_toy(env: PushCubeEnv_toy, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.2, 0, 0.05]), q=env.agent.tcp.pose.sp.q) 
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res

def solve(env: PushCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res