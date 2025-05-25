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
import uuid
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks import MicrowaveTaskEnv, MicrowaveTaskEnv_knife, MicrowaveTaskEnv_mug, MicrowaveTaskEnv_knife

def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type=None):
    error_description = {}
    gripper_state = -1
    speed_factor = 1.0
    objectA = "knife"
    objectB = "cup"
    if task_stage == "knife_reach":
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
        error_description["stage"] = "knife_reach"
        error_description["error_type"] = "position offset"
        # error_description["details"] = f"The robot arm did not align perfectly over {objectA}, causing a misalignment that prevented the robot from successfully reaching the position above the {objectA}. This misalignment can lead to further errors in subsequent stages of the task. Specifically, the misalignment may cause the robot to fail to grasp the knife properly, or grasp the knife in a suboptimal position, which could result in the knife being dropped in later stages of the task."
        # error_description["details"] = f"The robot arm did not align perfectly over {objectA}, causing a misalignment that prevented the robot from successfully reaching the position above the {objectA}. This misalignment can lead to further errors in subsequent stages of the task. Specifically, the misalignment may cause the robot to fail to grasp the knife properly."
        error_description["details"] = f"The robot arm did not align perfectly over {objectA}, causing a misalignment that prevented the robot from successfully reaching the position above the {objectA}. This misalignment can lead to further errors in subsequent stages of the task. Specifically, the misalignment cause the robot grasp the knife in a suboptimal position, which could result in the knife being dropped in later stages of the task."
        error_description["correction_suggestion"] = f"Adjust the reach position to align with {objectA} in the reach stage."

        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  # 方向向量
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "cup_reach":
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
        error_description["stage"] = "cup_reach"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The robot arm did not align perfectly over {objectB}"
        error_description["correction_suggestion"] = f"Adjust the reach position to align with {objectB} in the reach stage."
        # error_description["details"] = f"Position offset by {perturbation.tolist()}."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  # 方向向量
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "forget_knife":
        error_description["stage"] = "forget_knife"
        error_description["error_type"] = "step_omission"
        error_description["details"] = f"The knife was not placed into the {objectB} before putting the {objectB} into the microwave, causing the task to fail."
        error_description["correction_suggestion"] = f"Place the knife into the {objectB} before putting the {objectB} into the microwave."
    
    elif task_stage == "knife_grasp":
        if perturbation_type == "position_offset":
            gripper_state = -0.6
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
            error_description["stage"] = "knife_grasp"
            error_description["error_type"] = "position_offset"
            error_description["details"] = "The robot arm did not reach a suitable position for grasping."
            error_description["correction_suggestion"] = f"Adjust the grasp position to align with {objectA} in the grasp stage."
            # error_description["details"] = "Gripper closed before aligning with cubeA."
            
            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  # 方向向量
            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)
        elif perturbation_type == "gripper_error":
            gripper_state = -0.40
            speed_factor = 5.0
            error_description["stage"] = "knife_grasp"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = "The gripper did not grasp the cube tightly during the grasping stage."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force when reaching the grasp position to grasp the {objectA}."
            # error_description["details"] = "Gripper closed after passing cubeA."

    elif task_stage == "cup_grasp":
        if perturbation_type == "position_offset":
            gripper_state = -0.6

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
            error_description["stage"] = "cup_grasp"
            error_description["error_type"] = "position_offset"
            error_description["details"] = f"The robot arm did not reach a suitable position above the {objectB} during the reach stage, causing an improper and unstable grasp of the {objectB} in the grasp stage. This misalignment led to the knife being dropped and the subsequent failure of the task.",
            error_description["correction_suggestion"] = f"Adjust the grasp position to align with {objectB} in the grasp stage."
            # error_description["details"] = "Gripper closed before aligning with cubeA."
            
            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  # 方向向量
            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)
        elif perturbation_type == "gripper_error":
            gripper_state = 0.8
            speed_factor = 5.0
            error_description["stage"] = "cup_grasp"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The gripper did not close tightly enough to grasp the {objectB}, leading to the failure of subsequent tasks. This insufficient grip can cause the {objectB} to slip or be dropped during later stages, ultimately resulting in task failure."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force when reaching the grasp position to grasp the {objectA}."
            # error_description["details"] = "Gripper closed after passing cubeA."

    elif task_stage == "knife_upright":
        perturbation = np.random.uniform(-0.05, 0.05, size=3)  # Random position offset
        pose.p += perturbation
        error_description["stage"] = "knife_upright"
        error_description["error_type"] = "rotation offset"
        error_description["details"] = f"The {objectA} was not upright, causing it to fail to be placed into the {objectB}, leading to the task's final failure."
        error_description["correction_suggestion"] = f"Ensure the {objectA} is upright before attempting to place it into the {objectB}."
    
    elif task_stage == "knife_align":
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

        error_description["stage"] = "knife_align"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The knife was not aligned during the insertion stage, causing it to fail to be placed into the {objectB}, ultimately leading to the task failure."
        error_description["correction_suggestion"] = f"Adjust the alignment of the knife to ensure it is positioned over the {objectB} opening before attempting to place it into the {objectB}."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  # 方向向量
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "microwave_open":
        error_description["stage"] = "microwave_open"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The microwave door was not opened completely, causing the {objectB} to collide with the door and ultimately can not put the {objectB} in the microwave, leading to the task failure."
        error_description["correction_suggestion"] = f"Ensure the microwave door is fully open before attempting to place the {objectB} inside."
    
    elif task_stage == "microwave_close":
        error_description["stage"] = "microwave_close"
        error_description["error_type"] = "step_omission"
        error_description["details"] = f"The microwave door was not closed after putting the {objectB} inside, causing the task to fail."
        error_description["correction_suggestion"] = f"Close the microwave door after putting the {objectB} inside."

    return pose, gripper_state, error_description, speed_factor

def solve_with_errors(env: MicrowaveTaskEnv, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
    unique_id = str(uuid.uuid4())
    error_description = {}
    # logs = []
    is_print_logs = True
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

    # Introduce perturbations based on error_stage and perturbation_type
    obb = get_actor_obb(env.spoon)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    spoon_pose = env.spoon.pose
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

    if not error_stage == "forget_knife": 
        # Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        if error_stage == "knife_reach":
            reach_pose, _, error_description, _ = add_perturbation(error_stage, reach_pose)
        planner.move_to_pose_with_screw(reach_pose)
        # Grasp
        if error_stage == "knife_grasp":
            grasp_pose_, gripper_state, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
        else:
            grasp_pose_ = grasp_pose
            gripper_state = -0.6
        planner.move_to_pose_with_screw(grasp_pose_)
        planner.close_gripper(gripper_state=gripper_state)
        # Lift
        lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
        planner.move_to_pose_with_screw(lift_pose)
        lift_pose2 = lift_pose * sapien.Pose([0.18, 0, -0.1])
        planner.move_to_pose_with_screw(lift_pose2)
        # upright
        theta = np.pi * 0.225
        rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
        final_pose = lift_pose2 * sapien.Pose(
            p=[0, 0, 0],
            q=rotation_quat
        )
        if error_stage == "knife_upright":
            _, _, error_description, _ = add_perturbation(error_stage, final_pose)
            theta = 0
            rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
            final_pose = lift_pose2 * sapien.Pose(
                p=[0, 0, 0],
                q=rotation_quat
            )
        else:
            planner.move_to_pose_with_screw(final_pose)
        # Move to cup
        goal_pose = env.cup.pose * sapien.Pose([0, 0, 0.12])
        # offset = (goal_pose.p - env.spoon.pose.p).numpy()[0]
        # align_pose = sapien.Pose(lift_pose.p + offset, final_pose.q)
        align_pose = sapien.Pose(goal_pose.p.numpy()[0], final_pose.q)
        if error_stage == "knife_align":
            align_pose, _, error_description, _ = add_perturbation(error_stage, align_pose)
        planner.move_to_pose_with_RRTConnect(align_pose)
        align_pose2 = align_pose * sapien.Pose([-0.048, 0.0, 0.0])
        planner.move_to_pose_with_RRTConnect(align_pose2)
        planner.open_gripper()  
    # -----------------------------------------------  
    else: 
        _,_, error_description,_ = add_perturbation("forget_knife", grasp_pose)
        lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
        lift_pose2 = lift_pose * sapien.Pose([0.18, 0, -0.1])
        theta = np.pi * 0.225
        rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
        final_pose = lift_pose2 * sapien.Pose(
            p=[0, 0, 0],
            q=rotation_quat
        )
        goal_pose = env.cup.pose * sapien.Pose([0, 0, 0.12])
        align_pose = sapien.Pose(goal_pose.p.numpy()[0], final_pose.q)
        align_pose2 = align_pose * sapien.Pose([-0.048, 0.0, 0.0])
    lift_pose2 = align_pose2 * sapien.Pose([0.1, 0.0, 0.0])
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

    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.03])
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([0.03, 0.0, 0.0])
    planner.move_to_pose_with_screw(grasp_pose2)
    planner.close_gripper()

    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.5, -0.18, -0.1])
    offset = (goal_pose.p - env.cup.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
    # -----------------------------------------------

    lift_pose2 = align_pose * sapien.Pose([0.2, 0.0, 0.0])
    planner.move_to_pose_with_screw(lift_pose2)

    # pull the door of the microwave
    goal_pose = env.microwave.get_pose() * sapien.Pose([-0.1, -0.152, 0.18])
    offset = (goal_pose.p - align_pose2.p).numpy()[0]
    pull_pose_up = sapien.Pose(grasp_pose.p + offset, grasp_pose.q)
    planner.move_to_pose_with_screw(pull_pose_up)
    pull_pose = pull_pose_up * sapien.Pose([-0.05, 0.0, 0.0])
    planner.move_to_pose_with_screw(pull_pose)

    
    if error_stage == "microwave_open":
        _,_, error_description,_ = add_perturbation(error_stage, grasp_pose)

        pull_pose2 = pull_pose * sapien.Pose([0.0, -0.16, -0.03])
        planner.move_to_pose_with_screw(pull_pose2)
        pull_pose3 = pull_pose2 * sapien.Pose([0.0, -0.1, -0.05]) # 
        planner.move_to_pose_with_screw(pull_pose3)
    else:
        pull_pose2 = pull_pose * sapien.Pose([0.0, -0.16, -0.1])
        planner.move_to_pose_with_screw(pull_pose2)
        pull_pose3 = pull_pose2 * sapien.Pose([0.0, -0.1, -0.25]) # 
        planner.move_to_pose_with_screw(pull_pose3)

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

    reach_pose = grasp_pose * sapien.Pose([0.03, 0.0, -0.05])
    if error_stage == "cup_reach":
        reach_pose, _, error_description, _ = add_perturbation(error_stage, reach_pose)
    planner.move_to_pose_with_screw(reach_pose)
    grasp_pose2 = grasp_pose * sapien.Pose([-0.05, 0.0, 0.05])
    if error_stage == "cup_grasp":
        grasp_pose2, gripper_state, error_description, _ = add_perturbation(error_stage, grasp_pose2, perturbation_type=perturbation_type)
        planner.move_to_pose_with_screw(grasp_pose2)
        planner.close_gripper(gripper_state=gripper_state)
    else:
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

    depart_pose = move_pose2 * sapien.Pose([0, 0, -0.4])
    planner.move_to_pose_with_screw(depart_pose)

    if error_stage == "microwave_close":
        _,_, error_description,_ = add_perturbation(error_stage, grasp_pose)
    else:
        depart_pose2 = depart_pose * sapien.Pose([0, -0.2, 0])
        planner.move_to_pose_with_screw(depart_pose2)
        depart_pose3 = depart_pose2 * sapien.Pose([0, 0, 0.2])
        planner.move_to_pose_with_screw(depart_pose3)
        push_pose = depart_pose3 * sapien.Pose([0, 0.2, 0])
        planner.move_to_pose_with_screw(push_pose)
        push_pose2 = push_pose * sapien.Pose([0, 0, 0.2])
        planner.move_to_pose_with_screw(push_pose2)
    # --------------------------------------------------

    res = planner.close_gripper()
    planner.close()

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
        "MicrowaveTask", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(10):
        solve_with_errors(env, seed=seed,log_file='long_test', error_stage="forget_knife", debug=False, vis=True)