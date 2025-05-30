import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from tasks import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.task_PegInsertionSide import PegInsertionSideEnv_screwdriver

# def main():
#     env: PegInsertionSideEnv = gym.make(
#         "PegInsertionSide-v1",
#         obs_mode="none",
#         control_mode="pd_joint_pos",
#         render_mode="rgb_array",
#         reward_mode="dense",
#     )
#     for seed in range(100):
#         res = solve(env, seed=seed, debug=False, vis=True)
#         print(res[-1])
#     env.close()


def solve_peginsertionside_screwdriver(env: PegInsertionSideEnv_screwdriver, seed=None, debug=False, vis=False):
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
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()

    peg_init_pose = env.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([-max(0.05, env.peg_half_sizes[0, 0] / 2 + 0.01), 0, 0])
    grasp_pose = grasp_pose * (offset)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Align Peg
    # -------------------------------------------------------------------------- #

    # align the peg with the hole
    insert_pose = env.goal_pose * peg_init_pose.inv() * grasp_pose
    # insert_pose.q = torch.quaternion_multiply(torch.tensor([0, 0, 1, 0]), insert_pose.q)
    offset = sapien.Pose([-0.01 - env.peg_half_sizes[0, 0], 0, 0])
    pre_insert_pose = insert_pose * (offset)
    res = planner.move_to_pose_with_screw(pre_insert_pose)
    if res == -1: return res
    # refine the insertion pose
    for i in range(3):
        delta_pose = env.goal_pose * (offset) * env.peg.pose.inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        res = planner.move_to_pose_with_screw(pre_insert_pose)
        if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(insert_pose * (sapien.Pose([0.05, 0, 0])))
    if res == -1: return res
    planner.close()
    return res


# if __name__ == "__main__":
#     main()
