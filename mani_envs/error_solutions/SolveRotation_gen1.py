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
from tasks import SpinStackEnv, SpinStackEnv_gen1, SpinStackEnv_gen2
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
        solve_with_errors(env, log_file='rota_test', error_stage='early_grasp', perturbation_type="position_offset", seed=seed, debug=False, vis=True)

def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type="position_offset"):
    error_description = {}
    gripper_state = -1
    speed_factor = 1.0
    objectA = "apple"
    objectB = "orange"
    if task_stage == "cubeA_grasp":
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

            error_description["stage"] = f"{objectA}_grasp"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm failed to move to the correct position to grasp {objectA}, resulting in a failed grasp of {objectA}. Consequently, {objectA} could not be placed at the target position, and the subsequent stacking of {objectB} onto {objectA} also failed."
            error_description["correction_suggestion"] = f"Ensure the robot arm moves to the correct position to properly grasp {objectA}, allowing {objectA} to be placed at the target position and enabling the successful stacking of {objectB} onto {objectA}."
            
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
            error_description["stage"] = f"{objectA}_grasp"
            error_description["error_type"] = "rotation offset"
            error_description["details"] = f"The robot arm failed to grasp {objectA} in a suitable orientation, resulting in a failed grasp of {objectA}. Consequently, {objectA} could not be placed at the target position, and the subsequent stacking of {objectB} onto {objectA} also failed.",
            error_description["correction_suggestion"] = f"Ensure the robot arm grasps the {objectA} in a proper orientation, allowing {objectA} to be placed at the target position and enabling the successful stacking of {objectB} onto {objectA}."
            
            error_description["perturbation"] = rotation_angle.tolist() 
            error_description["perturbed_pose"] = pose.q.tolist()  
            error_description["direction_vector"] = rotation_quat.tolist()  

            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)
        
        elif perturbation_type == "gripper_error":
            gripper_state = -0.6
            speed_factor = 5.0
            error_description["stage"] = f"{objectA}_grasp"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The gripper did not grasp the {objectA} tightly during the grasping stage. Consequently, {objectA} could not be placed at the target position, and the subsequent stacking of {objectB} onto {objectA} also failed."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force when reaching the grasp position to grasp the {objectA}."
            # error_description["details"] = "Gripper closed after passing cubeA."
    elif task_stage == "cubeB_grasp":
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

            error_description["stage"] = f"{objectB}_grasp"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm failed to move to a proper position to grasp {objectB}, resulting in a failed grasp of {objectB}. Consequently, {objectB} could not be stacked onto {objectA}, leading to task failure."
            error_description["correction_suggestion"] = f"Ensure the robot arm grasps {objectB} in a proper position to successfully grasp {objectB} and enable it to be stacked onto {objectA}."

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
            error_description["stage"] = f"{objectB}_grasp"
            error_description["error_type"] = "rotation offset"
            error_description["details"] = f"The robot arm failed to grasp {objectB} in a suitable orientation, resulting in a failed grasp of {objectB}. Consequently, {objectB} could not be stacked onto {objectA}, leading to task failure."
            error_description["correction_suggestion"] = f"Ensure the robot arm grasps {objectB} in a proper orientation to successfully grasp {objectB} and enable it to be stacked onto {objectA}."
            
            error_description["perturbation"] = rotation_angle.tolist()  
            error_description["perturbed_pose"] = pose.q.tolist()  
            error_description["direction_vector"] = rotation_quat.tolist()  

            
            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)
        
        elif perturbation_type == "gripper_error":
            gripper_state = -0.6
            speed_factor = 5.0
            error_description["stage"] = f"{objectB}_grasp"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The gripper did not grasp the {objectB} tightly during the grasping stage. Consequently, {objectB} could not be stacked onto {objectA}, leading to task failure."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force when reaching the grasp position to grasp the {objectB}."
            # error_description["details"] = "Gripper closed after passing cubeA."

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
        error_description["details"] = f"The robot arm did not align {objectB} correctly over {objectA} while moving to the stacking position, causing the stacking of {objectB} onto {objectA} to fail."
        error_description["correction_suggestion"] = f"Adjust the robot arm's alignment to ensure {objectA} is correctly positioned over {objectB} before attempting to stack."
        # error_description["details"] = f"Misaligned stack position by {alignment_offset.tolist()}."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "early_grasp_cubeA":
        error_description["stage"] = f"{objectA}_grasp"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"During the rotation of {objectA} past the gripper, the robot arm failed to align with {objectA} and closed the gripper too early, resulting in a failed grasp of {objectA}. This caused {objectA} to not be placed stably under the turntable, leading to a failure to stack {objectB} onto {objectA} later in the task."
        error_description["correction_suggestion"] = f"During the rotation process, delay the grasping of {objectA}, ensure the robot arm aligns with {objectA}, and close the gripper at the appropriate time to successfully grasp {objectA}."
    
    elif task_stage == "late_grasp_cubeA":
        error_description["stage"] = f"{objectA}_grasp"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"During the rotation of {objectA} past the gripper, the robot arm failed to align with {objectA} and closed the gripper too late, resulting in a failed grasp of {objectA}. This caused {objectA} to not be placed stably under the turntable, leading to a failure to stack {objectB} onto {objectA} later in the task."
        error_description["correction_suggestion"] = f"During the rotation process, ensure the robot arm aligns with {objectA} and close the gripper at the appropriate time to successfully grasp {objectA}."
    
    elif task_stage == "early_grasp_cubeB":
        error_description["stage"] = f"{objectB}_grasp"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"During the rotation of {objectA} past the gripper, the robot arm failed to align with {objectA} and closed the gripper too early, resulting in a failed grasp of {objectA}. This caused {objectA} to not be placed stably under the turntable, leading to a failure to stack {objectB} onto {objectA} later in the task."
        error_description["correction_suggestion"] = f"During the rotation process, delay the grasping of {objectA}, ensure the robot arm aligns with {objectA}, and close the gripper at the appropriate time to successfully grasp {objectA}."
    
    elif task_stage == "late_grasp_cubeB":
        error_description["stage"] = f"{objectB}_grasp"
        error_description["error_type"] = "timing error"
        error_description["details"] = f"During the rotation of {objectA} past the gripper, the robot arm failed to align with {objectA} and closed the gripper too late, resulting in a failed grasp of {objectA}. This caused {objectA} to not be placed stably under the turntable, leading to a failure to stack {objectB} onto {objectA} later in the task."
        error_description["correction_suggestion"] = f"During the rotation process, ensure the robot arm aligns with {objectA} and close the gripper at the appropriate time to successfully grasp {objectA}."
    
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
        if error_stage == "early_grasp_cubeA":
            if angle > (np.pi * 1.5 - 0.65) and angle < (np.pi * 1.5 - 0.40):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                res = planner.move_to_pose_with_screw(grasp_pose)
                planner.close_gripper()
                break
        elif error_stage == "late_grasp_cubeA":
            if angle > (np.pi * 1.5 - 0.27) and angle < (np.pi * 1.5 - 0.25):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                res = planner.move_to_pose_with_screw(grasp_pose)
                planner.close_gripper()
                break
        elif angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            if error_stage == "cubeA_grasp":
                grasp_pose_, gripper_state, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
            else:
                gripper_state = -1
                grasp_pose_ = grasp_pose
            res = planner.move_to_pose_with_screw(grasp_pose_)
            planner.close_gripper(gripper_state=gripper_state)
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
        if error_stage == "early_grasp_cubeB":
            if angle > (np.pi * 1.5 - 0.70) and angle < (np.pi * 1.5 - 0.50):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                res = planner.move_to_pose_with_screw(grasp_pose)
                planner.close_gripper()
                break
        elif error_stage == "late_grasp_cubeB":
            if angle > (np.pi * 1.5 - 0.35) and angle < (np.pi * 1.5 - 0.30):
                _, _, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
                res = planner.move_to_pose_with_screw(grasp_pose)
                planner.close_gripper()
                break
        elif angle > (np.pi * 1.5 - 0.45) and angle < (np.pi * 1.5 - 0.42):
            if error_stage == "cubeB_grasp":
                grasp_pose_, gripper_state, error_description, _ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
            else:
                gripper_state = -1
                grasp_pose_ = grasp_pose
            res = planner.move_to_pose_with_screw(grasp_pose_)
            planner.close_gripper(gripper_state=gripper_state)
            break

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(lift_pose)
    # goal_pose = env.cube.pose * sapien.Pose([0, 0, 0.05]) 
    offset = (goal_pose.p - env.cubeB.pose.p).numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    if error_stage == "move_to_stack":
        align_pose, _, error_description, _ = add_perturbation(error_stage, align_pose, perturbation_type=perturbation_type)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()   
    
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