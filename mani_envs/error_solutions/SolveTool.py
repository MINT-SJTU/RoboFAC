import argparse
import gymnasium as gym
import numpy as np
import trimesh
import sapien
from transforms3d.euler import euler2quat
import torch # Add 
import time
# from mani_skill.examples.motionplanning.panda.error_solutions import add_perturbation_pull_cube_tool, add_perturbation_plug_charger

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb, check_collision, compute_grasp_pose)
from mani_skill.utils.wrappers.record import RecordEpisode
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks import ToolsTaskEnv
# from solve_tool_error import solveTools as solve_after_choose_tool
import uuid
import json
import time

def add_perturbation_pull_cube_tool(task_stage, pose, gripper_state=None, perturbation_type="position_offset"):
    error_description = {}
    speed_factor=1.0
    gripper_state = -1
    objectA = "charger"
    if task_stage == "tool_reach":
        gripper_state = 0.05
        if perturbation_type == "position_offset":
            perturbation = np.zeros(3)
            for i in range(3):
                if np.random.rand() > 0.5:
                    perturbation[i] = np.random.uniform(-0.07, -0.05)
                else:
                    perturbation[i] = np.random.uniform(0.05, 0.07)
            perturbation[2] = 0  # Only offset in X-Y plane
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation

            error_description["stage"] = "tool_reach"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm did not align well above the L-shaped tool, causing a bad grasp position and ultimately failing to use the tool to pull the {objectA}."
            error_description["correction_suggestion"] = f"Adjust the robot arm's alignment above the L-shaped tool to ensure a proper grasp position and successfully use the tool to pull the {objectA}."
            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)
        

        elif perturbation_type == "rotation_offset":
            rotation_angle = np.zeros(3)
            for i in range(3):
                    rotation_angle[i] = np.random.uniform(-0.3, 0.3) 
            # rotation_angle = np.random.uniform(-0.3, 0.3)  # Random rotation angle offset
            rotation_quat = euler2quat(rotation_angle[0], rotation_angle[1], rotation_angle[2])  # Apply rotation around Y-axis
            error_description["original_pose"] = pose.q.tolist()
            pose = (pose * sapien.Pose(q=rotation_quat))
            error_description["stage"] = "grasp"
            error_description["error_type"] = "rotation offset"
            error_description["details"] = f"The robot arm failed to grasp the L-shaped tool due to not having a suitable orientation, causing a task failure.",

            error_description["perturbation"] = rotation_angle.tolist() 
            error_description["perturbed_pose"] = pose.q.tolist() 
            error_description["direction_vector"] = rotation_quat.tolist()  

            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "choose_tool":
        error_description["stage"] = "choose_tool"
        error_description["error_type"] = "wrong_object"
        error_description["details"] = f"The robot arm selected the wrong tool for the task, causing a failure to pull the {objectA} to the target position."
        error_description["correction_suggestion"] = f"Ensure the robot arm selects the correct tool for the task to successfully pull the {objectA} to the target position."

    elif task_stage == "tool_grasp":
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

            error_description["stage"] = "tool_grasp"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm failed to grasp the L-shaped tool due to not aligning with the L-shaped tool, causing a task failure."
            error_description["correction_suggestion"] = f"Adjust the robot arm's alignment with the L-shaped tool to ensure a proper grasp and successfully complete the task."
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
            rotation_quat = euler2quat(rotation_angle[0], rotation_angle[1], rotation_angle[2])  # Apply rotation around Y-axis
            error_description["original_pose"] = pose.q.tolist()

            pose = (pose * sapien.Pose(q=rotation_quat))
            error_description["stage"] = "tool_grasp"
            error_description["error_type"] = "rotation offset"
            error_description["details"] = f"The robot arm grasped the L-shaped tool but not in a suitable orientation, causing the tool to slip during the lift phase and ultimately failing to use the tool to pull the {objectA}."
            error_description["correction_suggestion"] = f"Ensure the robot arm grasps the L-shaped tool in a suitable orientation to prevent slipping during the lift phase and successfully use the tool to pull the {objectA}."
            
            error_description["perturbation"] = rotation_angle.tolist()  
            error_description["perturbed_pose"] = pose.q.tolist()  
            error_description["direction_vector"] = rotation_quat.tolist() 

            
            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "lift_tool":
        if perturbation_type == "speed_offset":
            speed_factor = np.random.uniform(2.0, 5.0)
            # speed_factor = 8.0
            error_description["stage"] = "lift_tool"
            error_description["error_type"] = "gripper_error"
            error_description["details"] = f"The robot arm did not grasp the L-shaped tool tightly enough, and the excessive speed during the lift phase caused the tool to slip, ultimately failing to pull the {objectA}."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force and reduce the lifting speed to ensure the L-shaped tool is securely held and lifted properly."
    elif task_stage == "lower_tool":
        if perturbation_type == "position_offset":
            perturbation = np.zeros(3)
            for i in range(3):
                if np.random.rand() > 0.5:
                    perturbation[i] = np.random.uniform(-0.07, -0.05)
                else:
                    perturbation[i] = np.random.uniform(0.05, 0.07)
           
            perturbation[2] = 0  # Only offset in X-Y plane
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
            
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation
            error_description["stage"] = "lower_tool"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm did not move to a suitable position during the lower tool phase, ultimately failing to pull the {objectA} to the target position.",
            error_description["correction_suggestion"] = f"Adjust the robot arm's position during the lower tool phase to ensure proper alignment and successfully pull the {objectA} to the target position."
            
            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  
            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "tool_pull":
        direction_offset = np.random.uniform(0.1, 0.14, size=3)  # Random direction offset
        direction_offset[0] = 0  
        direction_offset[1] = 0  
        perturbation_magnitude = np.linalg.norm(direction_offset)
        if perturbation_magnitude > 0:
            direction_vector = direction_offset / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(direction_offset)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += direction_offset

        error_description["stage"] = "tool_pull"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The robot arm did not maintain the L-shaped tool at the same height and close to the table during the pull phase, causing the cube to slip and ultimately failing to pull the {objectA} to the target position.",
        error_description["correction_suggestion"] = f"Ensure the robot arm maintains the L-shaped tool at a consistent height and close to the table during the pull phase to prevent the cube from slipping and successfully pull the {objectA} to the target position."
        
        error_description["perturbation"] = direction_offset.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, direction_offset)]
        error_description["direction_description"] = ", ".join(direction_desc)

    return pose, speed_factor, error_description, gripper_state

def add_perturbation_plug_charger(task_stage, pose, gripper_state=None, perturbation_type=None):
    error_description = {}
    objectA = "charger"
    if task_stage == "plug_reach":
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                    perturbation[i] = np.random.uniform(-0.07, -0.05)
            else:
                    perturbation[i] = np.random.uniform(0.05, 0.07)
       
        perturbation[2] = 0  # Only offset in X-Y plane
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "plug_reach"
        error_description["error_type"] = "position offset"
        error_description["details"] = f"The robot arm did not reach a suitable position above the plug, causing it to fail to properly grasp the {objectA} during the grasp phase."
        error_description["correction_suggestion"] = f"Adjust the robot arm's trajectory to ensure it reaches the correct position above the plug before attempting to grasp the {objectA}."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "plug_grasp":
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

            error_description["stage"] = "plug_grasp"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm failed to grasp the L-shaped tool due to not aligning with the {objectA}, causing a task failure."
            error_description["correction_suggestion"] = f"Adjust the robot arm's alignment with the {objectA} to ensure a proper grasp and successfully complete the task."
            
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
            rotation_quat = euler2quat(rotation_angle[0], rotation_angle[1], rotation_angle[2])  # Apply rotation around Y-axis
            error_description["original_pose"] = pose.q.tolist()
            pose = (pose * sapien.Pose(q=rotation_quat))
            error_description["stage"] = "plug_grasp"
            error_description["error_type"] = "rotation offset"
            error_description["details"] = f"The robot arm grasped the plug but not in a good orientation, ultimately failing to insert the plug."
            error_description["correction_suggestion"] = f"Ensure the robot arm grasps the plug in a proper orientation to successfully align and insert it into the socket."
            
            error_description["perturbation"] = rotation_angle.tolist()  
            error_description["perturbed_pose"] = pose.q.tolist()  
            error_description["direction_vector"] = rotation_quat.tolist() 

            
            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "plug_align":
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

            error_description["stage"] = "plug_align"
            error_description["error_type"] = "position offset"
            error_description["details"] = f"The robot arm failed to align the plug with the socket due to a position offset. The misalignment caused the plug to be positioned incorrectly relative to the socket, resulting in an insertion failure."
            error_description["correction_suggestion"] = f"Adjust the robot arm's position to ensure the plug is correctly aligned with the socket before attempting insertion."            
           
            
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
            rotation_quat = euler2quat(rotation_angle[0], rotation_angle[1], rotation_angle[2])  # Apply rotation around Y-axis
            error_description["original_pose"] = pose.q.tolist()
            pose = (pose * sapien.Pose(q=rotation_quat))
            error_description["stage"] = "plug_align"
            error_description["error_type"] = "rotation offset"
            error_description["details"] = f"The robot arm failed to align the plug with the socket due to a rotation offset. The incorrect orientation of the plug caused it to be misaligned with the socket, leading to an insertion failure."
            error_description["correction_suggestion"] = f"Adjust the robot arm's orientation to ensure the plug is properly aligned with the socket before attempting insertion."
            error_description["perturbation"] = rotation_angle.tolist()   
            error_description["perturbed_pose"] = pose.q.tolist()  
            error_description["direction_vector"] = rotation_quat.tolist() 

            
            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "plug_insert":
        perturbation = np.zeros(3)
        for i in range(3):
            perturbation[i] = np.random.uniform(-0.05, 0.05)
       
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
        
        error_description["original_pose"] = pose.p.tolist()
        pose.p += perturbation

        error_description["stage"] = "plug_insert"
        error_description["error_type"] = "position offset"
        error_description["details"] = "The robot arm did not select the correct target position, resulting in a failure to insert into the socket."
        error_description["correction_suggestion"] = "Adjust the robot arm's trajectory to ensure the plug is inserted into the socket correctly."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
        
    return pose, gripper_state, error_description


def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

def solve_with_errors(env: ToolsTaskEnv, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
    unique_id = str(uuid.uuid4())
    logs = []
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

    grasp_pose = compute_grasp_pose(env, env.l_shape_tool, FINGER_LENGTH) # Get tool OBB and compute grasp pose
    gripper_state = -1
    if error_stage == "choose_tool":
        wrong_tool = np.random.choice([env.circular_shape_tool, env.y_shape_tool])
        grasp_pose = compute_grasp_pose(env, wrong_tool, FINGER_LENGTH)
        _, _, error_description, _ = add_perturbation_pull_cube_tool("choose_tool", grasp_pose)
        # res, unique_id = solve_after_choose_tool(env,grasp_pose=grasp_pose,vis=vis,unique_id=unique_id)
    else:
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
        speed_factor = 1.0
        if error_stage == "tool_reach": # add perturbation
            reach_pose, speed_factor, error_description, gripper_state = add_perturbation_pull_cube_tool("tool_reach", reach_pose, perturbation_type=perturbation_type)
        res = planner.move_to_pose_with_screw(reach_pose) # reach
        time.sleep(0.5)
        if res == -1: return res
        
        if error_stage == "tool_grasp":
            grasp_pose_, _, error_description,_ = add_perturbation_pull_cube_tool("tool_grasp", grasp_pose, perturbation_type=perturbation_type)
        else:
            grasp_pose_ = grasp_pose
        res = planner.move_to_pose_with_screw(grasp_pose_) # grasp
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper(gripper_state=gripper_state)

        lift_height = 0.35  
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        if error_stage == "lift_tool":
            lift_pose, speed_factor, error_description, _ = add_perturbation_pull_cube_tool("lift_tool", lift_pose, perturbation_type="speed_offset")
        res = planner.move_to_pose_with_screw(lift_pose, speed_factor=speed_factor) # lift_tool
        time.sleep(0.5)
        if res == -1: return res

        cube_pos = env.charger.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.width + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(approach_pose) # approching
        time.sleep(0.5)
        if res == -1: return res

        behind_offset = sapien.Pose(
            [-(env.width + env.cube_half_size),  
            -0.035,  
            0.02] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        if error_stage == "lower_tool":
            hook_pose_, speed_factor, error_description, _ = add_perturbation_pull_cube_tool("lower_tool", hook_pose, perturbation_type=perturbation_type)
        else:
            hook_pose_ = hook_pose
        res = planner.move_to_pose_with_screw(hook_pose_, speed_factor=speed_factor) # lower_tool
        time.sleep(0.5)
        if res == -1: return res

        # Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        if error_stage == "tool_pull":
            target_pose_, _, error_description, _ = add_perturbation_pull_cube_tool("tool_pull", target_pose)
        else:  
            target_pose_ = target_pose
        res = planner.move_to_pose_with_screw(target_pose_)
        time.sleep(0.5)
        if res == -1: return res

        push_pose = target_pose * sapien.Pose([0.1, 0.1, 0])
        res = planner.move_to_pose_with_screw(push_pose)
        time.sleep(0.5)
        if res == -1: return res

        planner.open_gripper()

        lift_pose = push_pose * sapien.Pose([0, 0, -0.2])
        res = planner.move_to_pose_with_screw(lift_pose)
        time.sleep(0.5)
        if res == -1: return res
        # sub_task: plug charger
        charger_base_pose = env.charger_base_pose
        charger_base_size = np.array(env.unwrapped._base_size) * 2

        obb = trimesh.primitives.Box(
            extents=charger_base_size,
            transform=charger_base_pose.sp.to_transformation_matrix(),
        )

        approaching = np.array([0, 0, -1])
        target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

        # add a angle to grasp
        grasp_angle = np.deg2rad(15)
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

        # -------------------------------------------------------------------------- #
        # Reach
        # -------------------------------------------------------------------------- #
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        if error_stage == "plug_reach":
            reach_pose,_,error_description = add_perturbation_plug_charger("plug_reach", reach_pose) # 添加一个随机扰动
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # -------------------------------------------------------------------------- #
        # Grasp
        # -------------------------------------------------------------------------- #
        if error_stage == "plug_grasp":
            grasp_pose,_,error_description = add_perturbation_plug_charger("plug_grasp", grasp_pose, perturbation_type=perturbation_type) # add perturbation
        planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        planner.close_gripper()
        # -------------------------------------------------------------------------- #
        # Align
        # -------------------------------------------------------------------------- #
        pre_insert_pose = (
            env.goal_pose.sp
            * sapien.Pose([-0.05, 0.0, 0.0])
            * env.charger.pose.sp.inv()
            * env.agent.tcp.pose.sp
        )
        insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp * sapien.Pose([0.0, 0.0, 0.0])
        if error_stage == "plug_align":
            pre_insert_pose,_,error_description = add_perturbation_plug_charger("plug_align", pre_insert_pose, perturbation_type=perturbation_type) # add perturbation
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0)
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=10)
        time.sleep(0.5)
        # -------------------------------------------------------------------------- #
        # Insert
        # -------------------------------------------------------------------------- #
        if error_stage == "plug_insert":
            insert_pose,_,error_description = add_perturbation_plug_charger("plug_insert", insert_pose)
        res = planner.move_to_pose_with_screw(insert_pose)
        time.sleep(0.5)
                
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

        "ToolsTask", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    for seed in range(20):
        # solveTools(env, seed=seed, debug=True, vis=True)
        # res = solve_with_errors(env,log_file='test',error_stage="lift_tool",perturbation_type="speed_offset", vis=True)
        res = solve_with_errors(env,log_file='tt',error_stage="tool_grasp",perturbation_type="rotation_offset", vis=True)
        time.sleep(1)













        