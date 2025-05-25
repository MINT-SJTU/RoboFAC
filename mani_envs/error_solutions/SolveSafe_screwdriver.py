import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import torch # Add 
import time
import json

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks import SafeTaskEnv, SafeTaskEnv_hammer, SafeTaskEnv_hammer, SafeTaskEnv_hammer, SafeTaskEnv_spatula
import uuid

def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type=None):
    error_description = {}
    gripper_state = -1
    speed_factor = 1.0
    objectA = "screwdriver"
    objectB = "safe"
    if task_stage == "screwdriver_reach":
        gripper_state = -0.40
        
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                perturbation[i] = np.random.uniform(-0.07, -0.05)
            else:
                perturbation[i] = np.random.uniform(0.05, 0.07)
        
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "screwdriver_reach"
        error_description["error_type"] = "position offset"
        # error_description["details"] = f"The robot arm did not align perfectly over {objectA}, causing a misalignment that prevented the robot from successfully reaching the position above the {objectA}. This misalignment can lead to further errors in subsequent stages of the task. Specifically, the misalignment may cause the robot to fail to grasp the spoon properly, or grasp the spoon in a suboptimal position, which could result in the spoon being dropped in later stages of the task."
        # error_description["details"] = f"The robot arm did not align perfectly over {objectA}, causing a misalignment that prevented the robot from successfully reaching the position above the {objectA}. This misalignment can lead to further errors in subsequent stages of the task. Specifically, the misalignment may cause the robot to fail to grasp the spoon properly."
        error_description["details"] = f"The robot arm did not align perfectly over {objectA}, causing a misalignment that prevented the robot from successfully reaching the position above the {objectA}. This misalignment can lead to further errors in subsequent stages of the task. Specifically, the misalignment cause the robot grasp the {objectA} in a suboptimal position, which could result in the {objectA} being dropped in later stages of the task."
        error_description["correction_suggestion"] = f"Adjust the reach position to align with {objectA} in the reach stage."
        
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
        

    elif task_stage == "safe_reach":
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                perturbation[i] = np.random.uniform(-0.07, -0.05)
            else:
                perturbation[i] = np.random.uniform(0.05, 0.07)
        
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "safe_reach"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"During the process of placing {objectA} into the safe, the robot arm did not reach a suitable position outside the safe, causing {objectA} to collide with the safe and fail to be placed inside successfully, resulting in task failure."
        error_description["correction_suggestion"] = f"Adjust the reach position of the robot arm to align with the safe before placing {objectA} inside."
        
        error_description["perturbation"] = perturbation.tolist()
        
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  

        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
        
    elif task_stage == "safe_knob_reach":
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                perturbation[i] = np.random.uniform(-0.07, -0.05)
            else:
                perturbation[i] = np.random.uniform(0.05, 0.07)
        
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation
        error_description["stage"] = "safe_knob_reach"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The robot arm did not align perfectly over {objectB}"
        error_description["correction_suggestion"] = f"Adjust the reach position to align with {objectB} in the reach stage."
        # error_description["details"] = f"Position offset by {perturbation.tolist()}."
        
        error_description["perturbation"] = perturbation.tolist()
        
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  

        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
        

    elif task_stage == "safe_insert":
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                perturbation[i] = np.random.uniform(-0.07, -0.05)
            else:
                perturbation[i] = np.random.uniform(0.05, 0.07)
        
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "safe_insert"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The robot arm did not place {objectA} into a suitable position inside the safe, causing {objectA} to not be placed inside the safe successfully, resulting in task failure."
        error_description["correction_suggestion"] = f"Adjust the position of the robot arm to ensure {objectA} is placed correctly inside the safe."
        
        error_description["perturbation"] = perturbation.tolist()
        
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  

        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "screwdriver_push":
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                perturbation[i] = np.random.uniform(-0.07, -0.05)
            else:
                perturbation[i] = np.random.uniform(0.05, 0.07)
        
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "screwdriver_push"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"After placing {objectA} into the safe, part of {objectA} was still outside. The robot arm failed to push it completely inside the safe, causing the safe door to not close and resulting in task failure."
        error_description["correction_suggestion"] = f"Ensure the robot arm pushes {objectA} completely inside the safe before attempting to close the door."
        
        error_description["perturbation"] = perturbation.tolist()
        
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  

        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "screwdriver_grasp":
        if perturbation_type == "position_offset":
            perturbation = np.random.uniform(-0.05, 0.05, size=3)  # Random position offset
            # perturbation = np.array([0.05, 0, 0])
            gripper_state = -0.6
            
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
            
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation

            error_description["stage"] = "screwdriver_grasp"
            error_description["error_type"] = "position_offset"
            error_description["details"] = f"The end effector of the robot arm did not align with {objectA}, causing it to fail to grasp {objectA}, resulting in task failure."
            error_description["correction_suggestion"] = f"Adjust the end effector position to align with {objectA} before attempting to grasp."
            
            error_description["perturbation"] = perturbation.tolist()
            
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  

            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)

        elif perturbation_type == "gripper_error":
            gripper_state = -0.40
            speed_factor = 5.0
            error_description["stage"] = "screwdriver_grasp"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The gripper did not grasp {objectA} tightly, causing it to slip and fail to be placed into the safe in subsequent stages."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force to ensure {objectA} is securely held before moving to the next stage."
    
    elif task_stage == "safe_knob_grasp":
        if perturbation_type == "position_offset":
            perturbation = np.random.uniform(-0.05, 0.05, size=3)  # Random position offset
            # perturbation = np.array([0.05, 0, 0])
            gripper_state = -0.6
            
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
            
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation

            error_description["stage"] = "safe_knob_grasp"
            error_description["error_type"] = "position_offset"
            error_description["details"] = f"The robot arm did not align with the safe knob during the grasping attempt, causing it to fail to grasp the safe knob and subsequently fail to rotate it, resulting in the safe not being locked and task failure."
            error_description["correction_suggestion"] = f"Adjust the robot arm's position to align with the safe knob before attempting to grasp and rotate it."
            
            
            error_description["perturbation"] = perturbation.tolist()
            
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  

            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)
        elif perturbation_type == "gripper_error":
            gripper_state = -0.40
            speed_factor = 5.0
            error_description["stage"] = "safe_knob_grasp"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The gripper did not grasp the {objectA} tightly during the grasping stage."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force when reaching the grasp position to grasp the {objectA}."
            # error_description["details"] = "Gripper closed after passing cubeA."

    elif task_stage == "safe_knob_rotate":
        error_description["stage"] = "safe_knob_rotate"
        error_description["error_type"] = "rotation offset"
        error_description["details"] = "The robot arm did not fully rotate the safe knob (did not rotate 180 degrees), causing the safe to not be locked properly and resulting in task failure."
        error_description["correction_suggestion"] = "Ensure the robot arm fully rotates the safe knob 180 degrees to lock the safe properly."
    
    elif task_stage == "safe_close":
        error_description["stage"] = "safe_close"
        error_description["error_type"] = "step_omission"
        error_description["details"] = F"The safe door was not closed after putting the {objectA} inside, causing the task to fail."
        error_description["correction_suggestion"] = f"Close the safe door after putting the {objectA} inside."

    return pose, gripper_state, error_description, speed_factor

def solve_with_errors(env: SafeTaskEnv, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
    unique_id = str(uuid.uuid4())
    error_description = {}
    gripper_state = -1
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
    offset = sapien.Pose([0.07, 0, 0.04])
    grasp_pose = grasp_pose * (offset)

    # Reach
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05]) 
    if error_stage == "screwdriver_reach":
        reach_pose, gripper_state, error_description, speed_factor = add_perturbation("screwdriver_reach", reach_pose, perturbation_type=perturbation_type)
        # print("hi")
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # Grasp
    if error_stage == "screwdriver_grasp":
        grasp_pose, gripper_state, error_description, speed_factor = add_perturbation("screwdriver_grasp", grasp_pose, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=gripper_state)
    # Lift to safe height
    lift_height = 0.2
    if error_stage == "screwdriver_lift":
        lift_height = 0.05
        error_description["stage"] = "screwdriver_lift"
        error_description["error_type"] = "insufficient lift height"
        error_description["details"] = f"Lift height reduced by {0.05}."

    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    safe_pose = env.safe.get_pose()
    safe_pose.set_q(grasp_pose.q)

    safe_reach_pose = safe_pose * sapien.Pose([0.3, 0.03, 0])
    if error_stage == "safe_reach":
        safe_reach_pose, gripper_state, error_description, speed_factor = add_perturbation("safe_reach", safe_reach_pose, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(safe_reach_pose)
    if res == -1: return res

    safe_move_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.08])
    if error_stage == "safe_insert":
        safe_move_pose, gripper_state, error_description, speed_factor = add_perturbation("safe_insert", safe_move_pose, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(safe_move_pose)
    if res == -1: return res
    planner.open_gripper()

    leave_pose1 = safe_pose * sapien.Pose([0.2, 0.03, 0.12])
    res = planner.move_to_pose_with_screw(leave_pose1)
    if res == -1: return res

    planner.close_gripper()
    push_pose = safe_pose * sapien.Pose([0.1, 0.03, 0.1])
    if error_stage == "screwdriver_push":
        push_pose, gripper_state, error_description, speed_factor = add_perturbation("screwdriver_push", push_pose, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(push_pose)
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
    if error_stage == "safe_close":
        _, _, error_description, _ = add_perturbation("safe_close", close_pose2, perturbation_type=perturbation_type)
    else:
        res = planner.move_to_pose_with_screw(close_pose2)
    if res == -1: return res
    res = planner.move_to_pose_with_screw(leave_pose2)
    if res == -1: return res

    planner.open_gripper()
    safe_pose_vertical = safe_pose_horizon * sapien.Pose(q=euler2quat(0, 0, np.pi * 0.5))

    grasp_pose1 = safe_pose_vertical * sapien.Pose([0.05, 0, -0.13])
    # grasp_pose1 = safe_pose_vertical * sapien.Pose([0.0, 0.05, -0.1])
    reach_pose1 = grasp_pose1 * sapien.Pose([0, 0, -0.1])
    if error_stage == "safe_knob_reach":
        reach_pose1, gripper_state, error_description, speed_factor = add_perturbation("safe_knob_reach", reach_pose1, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(reach_pose1)
    if res == -1: return res
    if error_stage == "safe_knob_grasp":
        grasp_pose1, gripper_state, error_description, speed_factor = add_perturbation("safe_knob_grasp", grasp_pose1, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(grasp_pose1)
    if res == -1: return res
    planner.close_gripper()

    rotation_pose1 = grasp_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
    if error_stage == "safe_knob_rotate":
        _, gripper_state, error_description, speed_factor = add_perturbation("safe_knob_rotate", rotation_pose1, perturbation_type=perturbation_type)
        rotation_pose1 = grasp_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi / 6))

    res = planner.move_to_pose_with_screw(rotation_pose1)
    if res == -1: return res
    if error_stage == "safe_knob_rotate":
        print("^-^")
    else:
        rotation_pose2 = rotation_pose1 * sapien.Pose(q=euler2quat(0, 0, - np.pi * 0.5))
        res = planner.move_to_pose_with_screw(rotation_pose2)
    if res == -1: return res

    res = planner.open_gripper()

    planner.close()
    # time.sleep(1)
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
    env = gym.make(
        "SafeTask-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(10):
        solve_with_errors(env, "log_test", "safe_knob_reach", perturbation_type="position_offset", seed=seed, vis=True)