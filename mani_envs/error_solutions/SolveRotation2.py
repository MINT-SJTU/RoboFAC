import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import torch # Add 
import time
import uuid
import json

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks import SpinPullStackEnv, SpinPullStackEnv_gen1, SpinPullStackEnv_gen2
from mani_skill.utils.structs.pose import Pose
def main(): 
    env = gym.make(
        "Rotation-v1",
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(10):
        solve_with_errors(env, log_file='rota_test', error_stage='early_pull_cubeA', perturbation_type="position_offset", seed=seed, debug=False, vis=True)

def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type="position_offset"):
    error_description = {}
    gripper_state = -1
    speed_factor = 1.0
    objectA = "green cube"
    objectB = "purple cube"
    if task_stage == "cubeA_grasp":
        gripper_state = -0.28
        if perturbation_type == "position_offset":
            perturbation = np.random.uniform(-0.07, 0.07, size=3)  # Random position offset
            perturbation[2] = 0  # Only offset in X-Y plane
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
           
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation

            error_description["stage"] = "cubeA_grasp"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm failed to move to the correct position to align with {objectA}, resulting in a failed grasp of {objectA}. Consequently, {objectA} could not be stacked onto {objectB}, leading to task failure."
            error_description["correction_suggestion"] = f"Ensure the robot arm moves to the correct position to properly align with {objectA}, allowing {objectA} to be successfully grasped and stacked onto {objectB}."

            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  
            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "cubeA_pull":
        perturbation = np.random.uniform(-0.2, 0.2, size=3)  # Random direction offset
        perturbation[2] = 0  # Only offset in X-Y plane
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
       
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "pull"
        error_description["error_type"] = "position offset"
        error_description["correction_suggestion"] = "Adjust the pull direction to align with the target."
        # error_description["details"] = f"Direction offset by {direction_offset.tolist()}."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "move_to_stack":
        # alignment_offset = np.random.uniform(-0.05, 0.05, size=3)
        perturbation = np.random.uniform(-0.12, 0.12, size=3)  
        # alignment_offset = np.random.uniform(0.08, 0.12, size=3)  
        perturbation[2] = 0 # Misalignment in X-Y plane
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
       
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "move_to_stack"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The robot arm did not align {objectA} correctly over {objectB} while moving to the stacking position, causing the stacking of {objectA} onto {objectB} to fail."
        error_description["correction_suggestion"] = f"Adjust the robot arm's alignment to ensure {objectA} is correctly positioned over {objectB} before attempting to stack."
        # error_description["details"] = f"Misaligned stack position by {alignment_offset.tolist()}."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "early_pull_cubeA":
        error_description["stage"] = "cubeA_pull"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"During the rotation of {objectA} to the front of the robot arm, the robot arm pulled {objectA} too early, failing to pull it under the turntable. Consequently, {objectA} continued moving on the turntable, causing the robot arm to fail to grasp the moving {objectA}. This ultimately resulted in a failure to stack {objectA} onto {objectB}."
        error_description["correction_suggestion"] = f"Delay the pulling action until {objectA} is properly aligned and stationary under the turntable. Ensure the robot arm pulls {objectA} at the correct time to enable successful grasping and stacking onto {objectB}."

    elif task_stage == "late_pull_cubeA":
        error_description["stage"] = "cubeA_pull"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"During the rotation of {objectA} to the front of the robot arm, the robot arm pulled {objectA} too late, failing to pull it under the turntable. Consequently, {objectA} continued moving on the turntable, causing the robot arm to fail to grasp the moving {objectA}. This ultimately resulted in a failure to stack {objectA} onto {objectB}."
        error_description["correction_suggestion"] = f"Ensure the pulling action is performed at the correct time when {objectA} is properly aligned and stationary under the turntable. This will enable successful grasping and stacking of {objectA} onto {objectB}."

    elif task_stage == "early_open_cubeA":
        error_description["stage"] = "open_cubeA"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"While preparing to stack {objectA} onto {objectB}, the robot arm released {objectA} too early before {objectB} had rotated into position beneath {objectA}. This caused {objectA} to fail to stack onto {objectB} successfully."
        error_description["correction_suggestion"] = f"Ensure the robot arm waits until {objectB} is properly aligned beneath {objectA} before releasing {objectA} to enable successful stacking."

    elif task_stage == "late_open_cubeA":
        error_description["stage"] = "open_cubeA"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"While preparing to stack {objectA} onto {objectB}, the robot arm released {objectA} too late after {objectB} had already rotated past the position beneath {objectA}. This caused {objectA} to fail to stack onto {objectB} successfully."
        error_description["correction_suggestion"] = f"Ensure the robot arm releases {objectA} at the correct time when {objectB} is properly aligned beneath {objectA} to enable successful stacking."
        
    return pose, gripper_state, error_description, speed_factor

def solve_with_errors(env, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    error_description = {}
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

    planner.close_gripper()

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
    reach_pose = grasp_pose * sapien.Pose([0.05, 0, 0])

    res = planner.move_to_pose_with_screw(reach_pose)
    pull_pose = grasp_pose * sapien.Pose([-0.18, 0, 0])
    

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cube.pose.sp.p[0] - env.disk_x) / (env.cube.pose.sp.p[1] - env.disk_y))
        if (env.cube.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle

        if error_stage == "early_pull_cubeA":
            if angle > (np.pi * 1.5 - 0.65) and angle < (np.pi * 1.5 - 0.40):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                res = planner.move_to_pose_with_screw(pull_pose)
                break
        elif error_stage == "late_pull_cubeA":
            if angle > (np.pi * 1.5 - 0.3) and angle < (np.pi * 1.5 - 0.2):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                res = planner.move_to_pose_with_screw(pull_pose)
                break
        elif angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            if error_stage == "cubeA_pull":
                pull_pose, gripper_state, error_description, _ = add_perturbation(error_stage, pull_pose, perturbation_type=perturbation_type)
            res = planner.move_to_pose_with_screw(pull_pose)
            break

    planner.open_gripper()
    is_cube_static = env.evaluate()["is_cube_static"]
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
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

    planner.move_to_pose_with_screw(reach_pose)
    if error_stage == "cubeA_grasp":
        grasp_pose, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    planner.move_to_pose_with_screw(lift_pose)


    goal_pose = Pose.create_from_pq(p=torch.tensor([env.disk_x - env.cubeB_r, env.disk_y, env.cubeB.pose.sp.p[2] + 0.1]), q=lift_pose.q)
    # offset = (goal_pose.p - env.cube.pose.p).numpy()[0]

    if is_cube_static:
        offset = (goal_pose.p - env.cube.pose.p).numpy()[0]
    else:

        tcp_pose = env.agent.tcp.pose.p  
        offset = (goal_pose.p - tcp_pose).numpy()[0]
    
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    if error_stage == "move_to_stack":
        align_pose, _, error_description, _ = add_perturbation(error_stage, align_pose, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(align_pose)

    for i in range(1000):
        planner.skip_step()
        angle = np.arctan((env.cubeB.pose.sp.p[0] - env.disk_x) / (env.cubeB.pose.sp.p[1] - env.disk_y))
        if (env.cubeB.pose.sp.p[1] - env.disk_y) < 0:
            angle = np.pi + angle

        if error_stage == "early_open_cubeA":
            if angle > (np.pi * 1.5 - 0.3) and angle < (np.pi * 1.5 - 0.09):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                planner.open_gripper()
                break
        elif error_stage == "late_open_cubeA":
            if angle > (np.pi * (-0.5) + 0.1) and angle < (np.pi * (-0.5) + 0.15):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                planner.open_gripper()
                break
        elif angle > (np.pi * 1.5 - 0.08) and angle < (np.pi * 1.5 - 0.02):
            planner.open_gripper()
            break
        if angle > (np.pi * 0.5 - 0.05) and angle < (np.pi * 0.5):
            res = planner.move_to_pose_with_screw(align_pose)

    for i in range(10):
        planner.skip_step()

    planner.close()

    unique_id = str(uuid.uuid4())
    success_level = "critical failure"
    log_entry = {
        "simulation_id": unique_id,
        "success_level": success_level,
        "error_description":error_description
    }

    with open(log_file, 'a') as f:
        json.dump(log_entry, f, indent=4)
        f.write('\n')

    return res, unique_id

if __name__ == "__main__":
    main()