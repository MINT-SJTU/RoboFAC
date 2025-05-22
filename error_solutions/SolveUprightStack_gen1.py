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
from tasks import UprightStackEnv

def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type="position_offset"):
    error_description = {}
    gripper_state = -1
    speed_factor = 1.0
    objectA = "long block"
    objectB = "bowl"
    if task_stage == "reach_peg":
        
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

        error_description["stage"] = "reach_peg"
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
        

    elif task_stage == "grasp_peg":
        gripper_state = -0.28
        if perturbation_type == "position_offset":
            perturbation = np.random.uniform(-0.05, 0.05, size=3)  # Random position offset
            perturbation[2] = 0  # Only offset in X-Y plane
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
            
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation

            error_description["stage"] = "grasp_peg"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm failed to grasp {objectA} due to not aligning with {objectA}, causing a task failure."
            error_description["correction_suggestion"] = f"Adjust the robot arm's alignment with {objectA} to ensure a proper grasp and successfully complete the task."
            
            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  
            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)
    
        elif perturbation_type == "rotation_offset":
            # rotation_angle = np.random.uniform(-0.3, 0.3)  # Random rotation angle offset

            rotation_angle = np.zeros(3)
            for i in range(3):
                rotation_angle[i] = np.random.uniform(-0.3, 0.3)

            rotation_quat = euler2quat(rotation_angle[0], rotation_angle[1], rotation_angle[2])  
            error_description["original_pose"] = pose.q.tolist()
            pose = (pose * sapien.Pose(q=rotation_quat))
            error_description["stage"] = "grasp_peg"
            error_description["error_type"] = "rotation offset"
            error_description["details"] = f"The robot arm did not grasp the {objectA} in a suitable orientation, causing it to fail to place the {objectA} upright and stably on the {objectB} during the stacking phase, resulting in task failure.",
            error_description["correction_suggestion"] = f"Ensure the robot arm grasps the {objectA} in a proper orientation to successfully place it upright and stably on the {objectB} during the stacking phase."
            
            error_description["perturbation"] = rotation_angle.tolist()  
            error_description["perturbed_pose"] = pose.q.tolist()  
            error_description["direction_vector"] = rotation_quat.tolist()  

            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)
        
        
        elif perturbation_type == "gripper_error":
            gripper_state = 0.15
            speed_factor = 5.0
            error_description["stage"] = "grasp_peg"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The gripper did not grasp the {objectA} tightly during the grasping stage, causing it to slip during the lift phase and ultimately failing to stack it upright on the {objectB}."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force when reaching the grasp position to grasp the {objectA}."
            # error_description["details"] = "Gripper closed after passing cubeA."

    elif task_stage == "lift_peg":
        speed_factor = 1.0
        if perturbation_type == "speed_offset":
            speed_factor = np.random.uniform(2.0, 5.0)
            # speed_factor = 8.0
            error_description["stage"] = "lift_peg"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The robot arm did not grasp {objectA} tightly enough, and during the lift phase, it moved too quickly, causing {objectA} to slip but not completely detach from the gripper. Consequently, it ultimately failed to stack {objectA} upright on {objectB}."
            error_description["correction_suggestion"] = f"Ensure the robot arm grasps {objectA} tightly and lifts it at an appropriate speed to prevent slipping and enable successful stacking onto {objectB}."
        
        elif perturbation_type == "position_offset":
            perturbation = np.zeros(3)
            perturbation[2] = np.random.uniform(-0.09, -0.08)
            # height_offset = 0.02
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
            
            error_description["original_pose"] = pose.p.tolist()
            pose.p[2] += perturbation[2]

            error_description["stage"] = "lift_peg"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm did not lift {objectA} to a sufficient height, causing it to fail to stack {objectA} onto {objectB} in subsequent steps."
            error_description["correction_suggestion"] = f"Ensure the robot arm lifts {objectA} to the required height to successfully stack it onto {objectB}."
            
            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  
            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "move_to_stack":
        # alignment_offset = np.random.uniform(-0.05, 0.05, size=3)
        perturbation = np.random.uniform(-0.2, +0.2, size=3) 
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

    elif task_stage == "place_upright":
        rotation_angle = np.zeros(3)
        for i in range(3):
               rotation_angle[i] = np.random.uniform(-0.7, 0.7) 
        rotation_quat = euler2quat(rotation_angle[0], rotation_angle[1], rotation_angle[2])  # Apply rotation around Y-axis
        pose = (pose * sapien.Pose(q=rotation_quat))
        error_description["stage"] = "place_upright"
        error_description["error_type"] = "rotation offset"
        error_description["details"] = f"The robot arm did not position {objectA} at a proper upright angle, causing {objectA} to fall over when released and fail to stand stably on {objectB}, resulting in task failure."
        error_description["correction_suggestion"] = f"Ensure the robot arm positions {objectA} at a proper upright angle before releasing it to ensure it stands stably on {objectB}."
        
        error_description["perturbation"] = rotation_angle.tolist() 
        error_description["perturbed_pose"] = pose.q.tolist()  
        error_description["direction_vector"] = rotation_quat.tolist()  

        directions = ["roll", "pitch", "yaw"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
        error_description["direction_description"] = ", ".join(direction_desc)
    return pose, gripper_state, error_description, speed_factor

def solve_with_errors(env: UprightStackEnv, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    unique_id = str(uuid.uuid4())
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
    if error_stage == "reach_peg":
        reach_pose, _, error_description, speed_factor = add_perturbation("reach_peg", reach_pose, perturbation_type=perturbation_type)
    res = planner.move_to_pose_with_screw(reach_pose)
    
    # Grasp
    if error_stage == "grasp_peg":
        grasp_pose_, gripper_state, error_description, speed_factor = add_perturbation("grasp_peg", grasp_pose, perturbation_type=perturbation_type)
    else:
        grasp_pose_ = grasp_pose
        gripper_state = -0.28
    res = planner.move_to_pose_with_screw(grasp_pose_)
    
    # planner.close_gripper(gripper_state=-0.2)
    planner.close_gripper(gripper_state=gripper_state)
    # Lift
    lift_pose = sapien.Pose([0.0, 0, 0.3]) * grasp_pose
    if error_stage == "lift_peg":
        lift_pose_, _, error_description, speed_factor = add_perturbation("lift_peg", lift_pose, perturbation_type=perturbation_type)
    else:
        lift_pose_ = lift_pose
        speed_factor = 1.0
    res = planner.move_to_pose_with_screw(lift_pose_, speed_factor=speed_factor)
    time.sleep(0.1)
    
    # Move
    target_pose = env.cubeA.pose * sapien.Pose([0.02, 0, 0.25])
    target_pose.q = lift_pose.q
    if error_stage == "move_to_stack":
        target_pose_, _, error_description, speed_factor = add_perturbation("move_to_stack", target_pose, perturbation_type=perturbation_type)
    else:
        target_pose_ = target_pose
    res = planner.move_to_pose_with_screw(target_pose_)
    time.sleep(0.1)
    
    # Place upright
    # theta = np.pi * 0.086
    theta = np.pi * 0.108
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    final_pose = target_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    if error_stage == "place_upright":
        final_pose_, _, error_description, speed_factor = add_perturbation("place_upright", final_pose, perturbation_type=perturbation_type)
    else:
        final_pose_ = final_pose
    res = planner.move_to_pose_with_screw(final_pose_)
    
    time.sleep(0.1)
    # Lower
    lower_pose =  env.cubeA.pose * sapien.Pose([0.02, 0, 0.155])
    lower_pose.q = final_pose.q
    res = planner.move_to_pose_with_screw(lower_pose)
    
    time.sleep(0.1)
    planner.open_gripper()
    time.sleep(1)

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
        "UprightStack-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(20):
        # solveTools(env, seed=seed, debug=True, vis=True)
        res, unique_id = solve_with_errors(env,log_file='error_log',error_stage="move_to_stack",perturbation_type="position_offset", vis=False)
        # res = solveUprightStack(env,vis=True)
        print(unique_id)
        time.sleep(1)