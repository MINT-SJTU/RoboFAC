import numpy as np
import sapien
import gymnasium as gym
import random
import uuid
import json
import time
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks import PullCubeToolEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.wrappers.record import RecordEpisode
import os.path as osp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.task_PullCubeTool import PullCubeToolEnv_golf, PullCubeToolEnv_dice, PullCubeToolEnv_can, PullCubeToolEnv_peach, PullCubeToolEnv_marble

from mani_skill.examples.motionplanning.panda.generalization.generalization.Choice_Error.choice_Error.task_PullCubeTool import PullCubeToolEnv_golf_more, PullCubeToolEnv_dice_more, PullCubeToolEnv_can_more, PullCubeToolEnv_peach_more, PullCubeToolEnv_marble_more

ERROR_DESCRIPTIONS_TEMPLATE = {
    "reach_failure": "The robot arm failed to reach above the {objectA}, causing a grasp failure.",
    "reach_bad_position": "The robot arm did not align well above the {objectA}, causing a bad grasp position and ultimately failing to use the tool to pull the {objectB}.",
    "reach_partial_success": "The robot arm did not align well above the {objectA}, causing a bad grasp position but managed to complete the task with difficulty.",
    
    "grasp_position_failure": "The robot arm failed to grasp the {objectA} due to not moving to a suitable position, causing a task failure.",
    "grasp_position_bad": "The robot arm grasped the {objectA} but not in a good position, ultimately failing to use the tool to pull the {objectB}.",
    "grasp_position_partial_success": "The robot arm grasped the {objectA} in a suboptimal position but managed to complete the task with difficulty.",
    
    "grasp_rotation_failure": "The robot arm failed to grasp the {objectA} due to not having a suitable orientation, causing a task failure.",
    "grasp_rotation_lift_failure": "The robot arm grasped the {objectA} but not in a suitable orientation, causing the tool to slip during the lift phase and ultimately failing to use the tool to pull the {objectB}.",
    "grasp_rotation_partial_success": "The robot arm grasped the {objectA} in a suboptimal orientation, causing the tool to slip during the lift phase but managed to complete the task with difficulty.",
    
    "lift_tool_failure": "The robot arm moved too fast during the lift phase, causing the {objectA} to slip and ultimately failing to use the tool to pull the {objectB}.",
    "lift_tool_partial_success": "The robot arm moved too fast during the lift phase, causing the {objectA} to slip but managed to complete the task with difficulty.",
    
    "lower_tool_collision_failure": "The robot arm did not move to a suitable position during the lower tool phase, causing the {objectA} to collide with the {objectB} and ultimately failing to pull the {objectB} to the target position.",
    "lower_tool_collision_partial_success": "The robot arm did not move to a suitable position during the lower tool phase, causing the {objectA} to collide with the {objectB} but managed to pull the {objectB} close to the target position with difficulty.",
    "lower_tool_position_failure": "The robot arm did not move to a suitable position during the lower tool phase, ultimately failing to pull the {objectB} to the target position.",
    "lower_tool_position_partial_success": "The robot arm did not move to a suitable position during the lower tool phase but managed to pull the {objectB} close to the target position with difficulty.",
    
    "pull_height_failure": "The robot arm did not maintain the {objectA} at the same height and close to the table during the pull phase, causing the {objectB} to slip and ultimately failing to pull the {objectB} to the target position.",
    "pull_height_partial_success": "The robot arm did not maintain the {objectA} at the same height and close to the table during the pull phase, causing the {objectB} to slip but managed to pull the {objectB} close to the target position with difficulty."
}
def get_error_descriptions(objectA, objectB):
    return {key: value.format(objectA=objectA, objectB=objectB) for key, value in ERROR_DESCRIPTIONS_TEMPLATE.items()}

ERROR_DESCRIPTIONS = get_error_descriptions(objectA="L-shape tool", objectB="peach")

def check_collision(env, entity1, entity2):
    contacts = env.scene.get_contacts()  
    # print(contacts)
    for contact in contacts:
        try:
            body1, body2 = contact.bodies
            if (body1.entity.name == "scene-0_"+entity1.name and body2.entity.name == "scene-0_"+entity2.name) or (body1.entity.name == "scene-0_"+entity2.name and body2.entity.name == "scene-0_"+entity1.name):

                return True
        except AttributeError as e:
            print(f"Error accessing contact bodies: {e}")
    return False

def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

# Add perturbation to specific subtasks
def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type="position_offset"):
    error_description = {}
    speed_factor=1.0
    objectA = "peach"
    if task_stage == "reach":
        if perturbation_type == "position_offset":
            perturbation = np.zeros(3)
            for i in range(3):
                if np.random.rand() > 0.5:
                    perturbation[i] = np.random.uniform(-0.08, -0.06)
                else:
                    perturbation[i] = np.random.uniform(0.06, 0.08)
         
            perturbation[2] = 0  # Only offset in X-Y plane
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
           
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation

            error_description["stage"] = "reach"
            error_description["error_type"] = "position offset"
           
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
           
            error_description["perturbation"] = rotation_angle.tolist() 
            error_description["perturbed_pose"] = pose.q.tolist() 
            error_description["direction_vector"] = rotation_quat.tolist()  

            
            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)
            
    elif task_stage == "grasp":
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
            error_description["stage"] = "grasp"
            error_description["error_type"] = "position offset"
            error_description["correction_suggestion"] = f"Adjust the robot arm's alignment with the L-shaped tool to ensure a proper grasp and successfully complete the task."
            
           
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
            rotation_quat = euler2quat(rotation_angle[0], rotation_angle[1], rotation_angle[2])  # Apply rotation around Y-axis
           
            error_description["original_pose"] = pose.q.tolist()
            pose = (pose * sapien.Pose(q=rotation_quat))
            error_description["stage"] = "grasp"
            error_description["error_type"] = "rotation offset"
           
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
            # error_description["details"] = f"The robot arm did not grasp the L-shaped tool tightly enough, and the excessive speed during the lift phase caused the tool to slip, ultimately failing to pull the {objectA}."
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
            error_description["correction_suggestion"] = f"Adjust the robot arm's position during the lower tool phase to ensure proper alignment and successfully pull the {objectA} to the target position."
           
            error_description["perturbation"] = perturbation.tolist()
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  
            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "pull":
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

        error_description["stage"] = "pull"
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


def solve_with_errors(env: PullCubeToolEnv_can, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
    # if error_stage == "none":
    #     solve_pullcubetool_peach(env, seed=seed, debug=debug, vis=vis)
    #     return
    error_description = {}
    env.reset(seed=seed)
    logs = []
    success_level = ""
    is_print_logs = True
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

    # Get tool OBB and compute grasp pose
    tool_obb = get_actor_obb(env.l_shape_tool)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        tool_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.l_shape_tool.pose.sp.p)
    offset = sapien.Pose([0.02, 0, 0])
    grasp_pose = grasp_pose * (offset)

    if error_stage == "reach":
        # (1) Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        reach_pose, speed_factor, error_description, _ = add_perturbation(error_stage, reach_pose, perturbation_type=perturbation_type)
        res = planner.move_to_pose_with_screw(reach_pose, speed_factor=speed_factor)
        if res == -1: return res
        time.sleep(0.5)
        # (2) Grasp
        res = planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.l_shape_tool) # Check if the tool is grasped
        time.sleep(0.5)
        # (3) Lift tool to safe height
        lift_height = 0.35  
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        res = planner.move_to_pose_with_screw(lift_pose,speed_factor=1.0)
        time.sleep(0.5)
        if res == -1: return res
        # (4) Approach cube
        cube_pos = env.cube.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.hook_length + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(approach_pose)
        time.sleep(0.5)
        if res == -1: return res
        # (5) Lower tool behind cube
        behind_offset = sapien.Pose(
            [-(env.hook_length + 4 * env.cube_half_size),  
            -0.067,  
            0] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(hook_pose,speed_factor=1.0)
        time.sleep(0.5)
        if res == -1: return res
        # (6) Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        res = planner.move_to_pose_with_screw(target_pose,speed_factor=1.0)
        if res == -1: return res
        # Evaluate task and record logs
        result_summary = env.evaluate()
        if not is_grasped:
            logs.append(ERROR_DESCRIPTIONS["reach_failure"]) 
            success_level = "critical failure"
        elif is_grasped and not result_summary["success"]:
            logs.append(ERROR_DESCRIPTIONS["reach_bad_position"])
            success_level = "critical failure"
        else:
            logs.append(ERROR_DESCRIPTIONS["reach_partial_success"])
            success_level = "partial success"
        if is_print_logs:
           print(logs)
    elif error_stage == "grasp":
        # (1) Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        speed_factor = 1.0
        res = planner.move_to_pose_with_screw(reach_pose, speed_factor=speed_factor)
        if res == -1: return res
        time.sleep(0.5)
        # (2) Grasp
        grasp_pose, _, error_description,_ = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type)
        res = planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.l_shape_tool) # Check if the tool is grasped
        time.sleep(0.5)
        # (3) Lift tool to safe height
        lift_height = 0.35  
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        res = planner.move_to_pose_with_screw(lift_pose,speed_factor=1.0)
        time.sleep(0.5)
        if res == -1: return res
        # (4) Approach cube
        cube_pos = env.cube.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.hook_length + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(approach_pose)
        time.sleep(0.5)
        if res == -1: return res
        # (5) Lower tool behind cube
        behind_offset = sapien.Pose(
            [-(env.hook_length + 4 * env.cube_half_size),  
            -0.067,  
            0] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(hook_pose,speed_factor=1.0)
        time.sleep(0.5)
        if res == -1: return res
        # (6) Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        res = planner.move_to_pose_with_screw(target_pose,speed_factor=1.0)
        if res == -1: return res
        # Evaluate task and record logs
        result_summary = env.evaluate()
        if perturbation_type == "position_offset":
            if not is_grasped:
                logs.append(ERROR_DESCRIPTIONS["grasp_position_failure"])
                success_level = "critical failure"
            elif is_grasped and not result_summary["success"]:
                logs.append(ERROR_DESCRIPTIONS["grasp_position_bad"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["grasp_position_partial_success"])
                success_level = "partial success"

        elif perturbation_type == "rotation_offset":
            if not is_grasped:
                logs.append(ERROR_DESCRIPTIONS["grasp_rotation_failure"])
                success_level = "critical failure"
            elif is_grasped and not result_summary["success"]:
                logs.append(ERROR_DESCRIPTIONS["grasp_rotation_lift_failure"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["grasp_rotation_partial_success"])
                success_level = "partial success"
        if is_print_logs:
            print(logs)
    elif error_stage == "lift_tool":
        # (1) Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return res
        time.sleep(0.5)
        # (2) Grasp
        res = planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.l_shape_tool) # Check if the tool is grasped
        time.sleep(0.5)
        # (3) Lift tool to safe height
        lift_height = 0.35  
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        lift_pose, speed_factor, error_description, _ = add_perturbation(error_stage, lift_pose, perturbation_type=perturbation_type)
        res = planner.move_to_pose_with_screw(lift_pose,speed_factor=speed_factor)
        time.sleep(0.5)
        if res == -1: return res
        # (4) Approach cube
        cube_pos = env.cube.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.hook_length + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(approach_pose)
        time.sleep(0.5)
        if res == -1: return res
        # (5) Lower tool behind cube
        behind_offset = sapien.Pose(
            [-(env.hook_length + 4 * env.cube_half_size),  
            -0.067,  
            0] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(hook_pose,speed_factor=1.0)
        time.sleep(0.5)
        if res == -1: return res
        # (6) Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        res = planner.move_to_pose_with_screw(target_pose,speed_factor=1.0)
        if res == -1: return res
        # Evaluate task and record logs
        result_summary = env.evaluate()
        if not result_summary["success"]:
            logs.append(ERROR_DESCRIPTIONS["lift_tool_failure"])
            success_level = "critical failure"
        else:
            logs.append(ERROR_DESCRIPTIONS["lift_tool_partial_success"])
            success_level = "partial success"
        if is_print_logs:
            print(logs)

    elif error_stage == "approching_cube":
        # (1) Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return res
        time.sleep(0.5)
        # (2) Grasp
        res = planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.l_shape_tool) # Check if the tool is grasped
        time.sleep(0.5)
        # (3) Lift tool to safe height
        lift_height = 0.35  
        speed_factor = 1.0
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        res = planner.move_to_pose_with_screw(lift_pose,speed_factor=speed_factor)
        time.sleep(0.5)
        if res == -1: return res
        # (4) Approach cube
        cube_pos = env.cube.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.hook_length + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        approach_pose, _, error_description, _ = add_perturbation(error_stage, approach_pose) # Add perturbation
        
        res = planner.move_to_pose_with_screw(approach_pose)
        time.sleep(0.5)
        if res == -1: return res
        # (5) Lower tool behind cube
        behind_offset = sapien.Pose(
            [-(env.hook_length + 4 * env.cube_half_size),  
            -0.067,  
            0] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(hook_pose,speed_factor=1.0)
        time.sleep(0.5)
        if res == -1: return res
        # (6) Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        res = planner.move_to_pose_with_screw(target_pose,speed_factor=1.0)
        if res == -1: return res
    elif error_stage == "lower_tool":
        # (1) Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return res
        time.sleep(0.5)
        # (2) Grasp
        res = planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.l_shape_tool) # Check if the tool is grasped
        time.sleep(0.5)
        # (3) Lift tool to safe height
        lift_height = 0.35  
        speed_factor = 1.0
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        res = planner.move_to_pose_with_screw(lift_pose,speed_factor=speed_factor)
        time.sleep(0.5)
        if res == -1: return res
        # (4) Approach cube
        cube_pos = env.cube.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.hook_length + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(approach_pose)
        time.sleep(0.5)
        if res == -1: return res
        # (5) Lower tool behind cube
        behind_offset = sapien.Pose(
            [-(env.hook_length + 4 * env.cube_half_size),  
            -0.067,  
            0] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        speed_factor = 1.0
        hook_pose, speed_factor, error_description, _ = add_perturbation(error_stage, hook_pose, perturbation_type=perturbation_type)
        # is_collision = check_collision(env, env.l_shape_tool, env.cube)
        # print(is_collision)
        res = planner.move_to_pose_with_screw(hook_pose,speed_factor=speed_factor)
        is_collision = check_collision(env, env.l_shape_tool, env.cube)
        # print(is_collision)
        time.sleep(0.5)
        if res == -1: return res
        # (6) Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        res = planner.move_to_pose_with_screw(target_pose,speed_factor=1.0)
        if res == -1: return res
        # Evaluate task and record logs
        result_summary = env.evaluate()
        if is_collision:
            if not result_summary["success"]:
                logs.append(ERROR_DESCRIPTIONS["lower_tool_collision_failure"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["lower_tool_collision_partial_success"])
                success_level = "partial success"
        else:
            if not result_summary["success"]:
                logs.append(ERROR_DESCRIPTIONS["lower_tool_position_failure"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["lower_tool_position_partial_success"])
                success_level = "partial success"

        if is_print_logs:
            print(logs)
        
    elif error_stage == "pull":
        # (1) Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return res
        time.sleep(0.5)
        # (2) Grasp
        res = planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.l_shape_tool) # Check if the tool is grasped
        time.sleep(0.5)
        # (3) Lift tool to safe height
        lift_height = 0.35  
        speed_factor = 1.0
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        res = planner.move_to_pose_with_screw(lift_pose,speed_factor=speed_factor)
        time.sleep(0.5)
        if res == -1: return res
        # (4) Approach cube
        cube_pos = env.cube.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.hook_length + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(approach_pose)
        time.sleep(0.5)
        if res == -1: return res
        # (5) Lower tool behind cube
        behind_offset = sapien.Pose(
            [-(env.hook_length + 4 * env.cube_half_size),  
            -0.067,  
            0] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        speed_factor = 1.0
        res = planner.move_to_pose_with_screw(hook_pose,speed_factor=speed_factor)
        time.sleep(0.5)
        if res == -1: return res
        # (6) Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        target_pose, _, error_description, _ = add_perturbation(error_stage, target_pose)
        res = planner.move_to_pose_with_screw(target_pose,speed_factor=1.0)
        if res == -1: return res
        # Evaluate task and record logs
        result_summary = env.evaluate()
        if not result_summary["success"]:
            logs.append(ERROR_DESCRIPTIONS["pull_height_failure"])
            success_level = "critical failure"
        else:
            logs.append(ERROR_DESCRIPTIONS["pull_height_partial_success"])
            success_level = "partial success"
        if is_print_logs:
            print(logs)

    elif error_stage == "none":
        # (1) Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return res
        time.sleep(0.5)
        # (2) Grasp
        res = planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        if res == -1: return res
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.l_shape_tool) # Check if the tool is grasped
        time.sleep(0.5)
        # (3) Lift tool to safe height
        lift_height = 0.35  
        speed_factor = 1.0
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
        res = planner.move_to_pose_with_screw(lift_pose,speed_factor=speed_factor)
        time.sleep(0.5)
        if res == -1: return res
        # (4) Approach cube
        cube_pos = env.cube.pose.sp.p
        approach_offset = sapien.Pose(
            [-(env.hook_length + env.cube_half_size + 0.08),  
            -0.0,  
            lift_height - 0.05]  
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        
        res = planner.move_to_pose_with_screw(approach_pose)
        time.sleep(0.5)
        if res == -1: return res
        # (5) Lower tool behind cube
        behind_offset = sapien.Pose(
            [-(env.hook_length + 4 * env.cube_half_size),  
            -0.067,  
            0] 
        )
        hook_pose = sapien.Pose(cube_pos) * behind_offset
        hook_pose.set_q(grasp_pose.q)
        speed_factor = 1.0
        res = planner.move_to_pose_with_screw(hook_pose,speed_factor=speed_factor)
        time.sleep(0.5)
        if res == -1: return res
        # (6) Pull cube
        pull_offset = sapien.Pose([-0.35, 0, 0])
        target_pose = hook_pose * pull_offset
        target_pose, _, error_description, _ = add_perturbation(error_stage, target_pose)
        res = planner.move_to_pose_with_screw(target_pose,speed_factor=1.0)
        if res == -1: return res

        success_level = "critical failure"
        logs.append("In front of the robot arm, there were an l-shape tool, a golf ball, a lego, and a can. The task was to use the l-shape tool to hook the lego to the front of the robot arm for easy grasping, but the robot arm hooked the golf ball instead, leading to the task's failure.")
        error_description["error_type"] = "wrong_object"

    planner.close()
    # 生成唯一标识符
    simulation_id = str(uuid.uuid4())
    error_description["details"] = logs[0]
    log_entry = {
        "simulation_id": simulation_id,
        "success_level": success_level,
        # "details": logs,
        "error_description":error_description
    }
    # 将日志写入文件
    with open(log_file, 'a') as f:
        json.dump(log_entry, f, indent=4)
        f.write('\n')
    return res, simulation_id

def solve_pullcubetool_peach(env: PullCubeToolEnv_peach, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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

    # Get tool OBB and compute grasp pose
    tool_obb = get_actor_obb(env.l_shape_tool)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        tool_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.l_shape_tool.pose.sp.p)
    offset = sapien.Pose([0.02, 0, 0])
    grasp_pose = grasp_pose * (offset)

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
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift tool to safe height
    # -------------------------------------------------------------------------- #
    lift_height = 0.35  
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    cube_pos = env.cube.pose.sp.p
    approach_offset = sapien.Pose(
        [-(env.hook_length + env.cube_half_size + 0.08),  
        -0.0,  
        lift_height - 0.05]  
    )
    approach_pose = sapien.Pose(cube_pos) * approach_offset
    approach_pose.set_q(grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(approach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower tool behind cube
    # -------------------------------------------------------------------------- #
    behind_offset = sapien.Pose(
        [-(env.hook_length + 4 * env.cube_half_size),   # 修改位置
        -0.067,  
        0] 
    )
    hook_pose = sapien.Pose(cube_pos) * behind_offset
    hook_pose.set_q(grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(hook_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Pull cube
    # -------------------------------------------------------------------------- #
    pull_offset = sapien.Pose([-0.35, 0, 0])
    target_pose = hook_pose * pull_offset
    res = planner.move_to_pose_with_screw(target_pose)
    if res == -1: return res

    planner.close()
    return res

if __name__ == '__main__':
    # quickstart
    env = gym.make(
        "PullCubeTool-peach-more", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    env = RecordEpisode(
        env,
        output_dir=osp.join('test', "motionplanning"),
        trajectory_name='test', save_video=True,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        save_on_reset=False
    )
    for seed in range(10):
        # solve(env,seed=seed,vis=True)
        # error_stage: reach, grasp, lift_tool,approching_cube, lower_tool, pull
        res = solve_with_errors(env,log_file='new',error_stage='none',perturbation_type="position_offset", seed=seed, debug=False, vis=True)
        # print(res)
        # env.render()
        # env.flush_trajectory()
        # env.flush_video(name='pull_cube_tool_01')
        time.sleep(1)
    env.close()