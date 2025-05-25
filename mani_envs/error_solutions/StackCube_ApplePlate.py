import json
import numpy as np
import sapien
from transforms3d.euler import euler2quat
from mani_skill.utils.wrappers.record import RecordEpisode
import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import torch 


import uuid
import time
from mani_skill.examples.motionplanning.panda.generalization.generalization.Choice_Error.choice_Error.task_StackCube import StackCubeEnv_appleplate_more, StackCubeEnv_can_more, StackCubeEnv_lego_more, StackCubeEnv_orangebowl_more, StackCubeEnv_dicebrick_more
from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.task_StackCube import StackCubeEnv_appleplate, StackCubeEnv_can, StackCubeEnv_lego, StackCubeEnv_orangebowl, StackCubeEnv_dicebrick

# Helper function to save error logs
# Error description templates
ERROR_DESCRIPTIONS_TEMPLATE = {
    "reach_partial_success": "The robot arm did not align perfectly over {objectA} during the reach phase but managed to grasp it successfully.",
    "reach_failure": "The robot arm failed to align over {objectA} during the reach phase, causing a grasp failure.",
    "grasp_partial_success": "The robot arm grasped {objectA} but not in the optimal position, potentially impacting later phases.",
    "grasp_failure": "The robot arm failed to grasp {objectA} due to an incorrect gripper position, missing the object's center of mass.",
    "lift_partial_success": "The robot arm initially lifted {objectA} but did not reach the optimal height, potentially leading to collisions in the stack phase.",
    "lift_failure": "The robot arm failed to lift {objectA} to the required height, causing a collision with {objectB}. Finally, it failed to stack {objectA} stably on {objectB} during the stack stage.",
    "stack_offset": "{objectA} was stacked on {objectB} but with a significant offset, causing the center of mass to shift beyond the support area.",
    "stack_failure": "{objectA} failed to stack on {objectB} due to misalignment.",
    "grasp_lift_failure": "The gripper did not grasp the {objectA} tightly during the grasping stage, causing the robot arm to fail to lift the {objectA} during the lift stage."
}

def get_error_descriptions(objectA, objectB):
    return {key: value.format(objectA=objectA, objectB=objectB) for key, value in ERROR_DESCRIPTIONS_TEMPLATE.items()}

ERROR_DESCRIPTIONS = get_error_descriptions(objectA="apple", objectB="plate")

def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

# Add perturbation to specific subtasks
def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type=None):
    error_description = {}
    gripper_state = -1
    speed_factor = 1.0
    objectA = "apple"
    objectB = "plate"
    if task_stage == "reach":
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

        error_description["stage"] = "reach"
        error_description["error_type"] = "position offset"
        error_description["root_cause"] = f"The robot arm did not align perfectly over {objectA}"
        error_description["correction_suggestion"] = f"Adjust the reach position to align with {objectA} in the reach stage."
        # error_description["details"] = f"Position offset by {perturbation.tolist()}."
                
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
       

    elif task_stage == "grasp":
        if perturbation_type == "position_offset":
            perturbation = np.random.uniform(-0.05, 0.05, size=3)  # Random position offset
           
            perturbation_magnitude = np.linalg.norm(perturbation)
            if perturbation_magnitude > 0:
                direction_vector = perturbation / perturbation_magnitude
            else:
                direction_vector = np.zeros_like(perturbation)
           
            error_description["original_pose"] = pose.p.tolist()
            pose.p += perturbation
            error_description["stage"] = "grasp"
            error_description["error_type"] = "position_offset"
            error_description["root_cause"] = "The robot arm did not reach a suitable position for grasping."
            error_description["correction_suggestion"] = f"Adjust the grasp position to align with {objectA} in the grasp stage."
            
            error_description["perturbation"] = perturbation.tolist()
            
            error_description["perturbed_pose"] = pose.p.tolist()
            error_description["direction_vector"] = direction_vector.tolist()  

            
            directions = ["x", "y", "z"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
            error_description["direction_description"] = ", ".join(direction_desc)
        elif perturbation_type == "gripper_error":
            gripper_state = 0.85
            speed_factor = 5.0
            error_description["stage"] = "grasp"
            error_description["error_type"] = "gripper_error"
            error_description["root_cause"] = f"The gripper did not grasp the {objectA} tightly during the grasping stage."
            error_description["correction_suggestion"] = f"Increase the gripper's grasping force when reaching the grasp position to grasp the {objectA}."
            # error_description["details"] = "Gripper closed after passing cubeA."

    elif task_stage == "lift":
        # height_offset = np.random.uniform(0.001, 0.02)
        # height_offset = -0.08  # Reduce lift height
        # pose = sapien.Pose([0, 0, height_offset]) * pose
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

        error_description["stage"] = "lift"
        error_description["error_type"] = "position offset"
        error_description["root_cause"] = f"The robot arm did not lift {objectA} to a sufficient height, causing it to fail to stack {objectA} onto {objectB} in subsequent steps."
        error_description["correction_suggestion"] = "Adjust the lift position to reach the required height in the lift stage."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "stack":
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                perturbation[i] = np.random.uniform(-0.12, -0.08)
            else:
                perturbation[i] = np.random.uniform(0.08, 0.12)
        perturbation[2] = 0 # Misalignment in X-Y plane
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude > 0:
            direction_vector = perturbation / perturbation_magnitude
        else:
            direction_vector = np.zeros_like(perturbation)
       
        error_description["original_pose"] = pose.p.tolist()
        pose.p[:2] = pose.p[:2] + perturbation  

        error_description["stage"] = "stack"
        error_description["error_type"] = "position offset"
        error_description["root_cause"] = f"The robot arm did not align {objectA} correctly over {objectB}."
        error_description["correction_suggestion"] = f"Adjust the stack position to align {objectA} over {objectB} in the stack stage."
        # error_description["details"] = f"Misaligned stack position by {perturbation.tolist()}."
        
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)

    return pose, gripper_state, error_description, speed_factor

# Main solve function with added perturbations
def solve_with_errors(env: StackCubeEnv_appleplate, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    logs = []
    success_level = ""
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
    obb = get_actor_obb(env.cubeA)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    flag1 = False
    flag2 = False
    flag3 = True
    flag4 = False

    # Reach
    if error_stage == 'reach':
        # sub_stage1: reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        reach_pose, _, error_description, speed_factor = add_perturbation("reach", reach_pose)
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # sub_stage2: grasp
        planner.move_to_pose_with_screw(grasp_pose)
        time.sleep(0.5)
        planner.close_gripper()
        time.sleep(0.5)
        # sub_stage3: lift
        lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
        planner.move_to_pose_with_screw(lift_pose)
        is_grasp = env.evaluate()["is_cubeA_grasped"]
        time.sleep(0.5)
        # sub_stage4: stack
        goal_pose = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 4])
        stack_pose = goal_pose
        # Align the stack pose
        offset = (goal_pose.p - env.cubeA.pose.p).numpy()[0]  
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        planner.move_to_pose_with_screw(align_pose)
        time.sleep(0.5)

        res = planner.open_gripper()
        time.sleep(1)

        result = env.evaluate()
        if result["is_cubeA_on_cubeB"] and is_grasp:
            logs.append(ERROR_DESCRIPTIONS["reach_partial_success"])
            success_level = "partially success"
        elif not result["is_cubeA_on_cubeB"] and not is_grasp:
            logs.append(ERROR_DESCRIPTIONS["reach_failure"])
            success_level = "critical failure"

    elif error_stage == 'grasp':
        # sub_stage1: reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # sub_stage2: grasp
        grasp_pose, gripper_state, error_description, speed_factor = add_perturbation("grasp", grasp_pose, perturbation_type=perturbation_type)
        planner.move_to_pose_with_screw(grasp_pose) # error
        time.sleep(0.5)
        planner.close_gripper(gripper_state=gripper_state)
        time.sleep(0.5)
        # sub_stage3: lift
        lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
        planner.move_to_pose_with_screw(lift_pose, speed_factor=speed_factor)
        is_grasp = env.evaluate()["is_cubeA_grasped"]
        time.sleep(0.5)
        # sub_stage4: stack
        goal_pose = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 4])
        stack_pose = goal_pose
        # Align the stack pose
        offset = (goal_pose.p - env.cubeA.pose.p).numpy()[0]  
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        planner.move_to_pose_with_screw(align_pose)
        time.sleep(0.5)

        res = planner.open_gripper()
        time.sleep(1)

        result = env.evaluate()
        if perturbation_type == "position_offset":
            if result["is_cubeA_on_cubeB"] and is_grasp:
                logs.append(ERROR_DESCRIPTIONS["grasp_partial_success"])
                success_level = "partially success"
            elif not result["is_cubeA_on_cubeB"] and not is_grasp:
                logs.append(ERROR_DESCRIPTIONS["grasp_failure"])
                success_level = "critical failure"

        elif perturbation_type == "gripper_error":
                logs.append(ERROR_DESCRIPTIONS["grasp_lift_failure"])
                success_level = "critical failure"

        
    elif error_stage == 'lift':
        # sub_stage1: reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # sub_stage2: grasp
        planner.move_to_pose_with_screw(grasp_pose) 
        time.sleep(0.5)
        # planner.close_gripper(gripper_state=0.2)
        planner.close_gripper()
        time.sleep(0.5)
        # sub_stage3: lift
        lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
        lift_pose, _, error_description, speed_factor = add_perturbation("lift", lift_pose, perturbation_type=perturbation_type) # error
        planner.move_to_pose_with_screw(lift_pose, speed_factor=speed_factor)
        is_grasp = env.evaluate()["is_cubeA_grasped"]
        time.sleep(0.5)
        # sub_stage4: stack
        goal_pose = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 4])
        stack_pose = goal_pose
        # Align the stack pose
        offset = (goal_pose.p - env.cubeA.pose.p).numpy()[0] 
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        planner.move_to_pose_with_screw(align_pose)
        time.sleep(0.5)

        res = planner.open_gripper()
        time.sleep(1)

        result = env.evaluate()
        if result["is_cubeA_on_cubeB"]:
            logs.append(ERROR_DESCRIPTIONS["lift_partial_success"])
            success_level = "partially success"
        elif not result["is_cubeA_on_cubeB"]:
            logs.append(ERROR_DESCRIPTIONS["lift_failure"])
            success_level = "critical failure"
        
    elif error_stage == 'stack':
        # sub_stage1: reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # sub_stage2: grasp
        planner.move_to_pose_with_screw(grasp_pose) 
        time.sleep(0.5)
        planner.close_gripper()
        time.sleep(0.5)
        # sub_stage3: lift
        lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
        planner.move_to_pose_with_screw(lift_pose)
        is_grasp = env.evaluate()["is_cubeA_grasped"]
        time.sleep(0.5)
        # sub_stage4: stack
        goal_pose = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 4])
        goal_pose, _, error_description, speed_factor = add_perturbation("stack", goal_pose)
        # Align the stack pose
        offset = (goal_pose.p - env.cubeA.pose.p).numpy()[0]  
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        planner.move_to_pose_with_screw(align_pose)
        time.sleep(0.5)

        res = planner.open_gripper()
        time.sleep(1)

        result = env.evaluate()
        if result["is_cubeA_on_cubeB"]:
            logs.append(ERROR_DESCRIPTIONS["stack_offset"])
            success_level = "partially success"
        elif not result["is_cubeA_on_cubeB"]:
            logs.append(ERROR_DESCRIPTIONS["stack_failure"])
            success_level = "critical failure"

    elif error_stage == 'none':
        error_description = {}
        # sub_stage1: reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # sub_stage2: grasp
        planner.move_to_pose_with_screw(grasp_pose) 
        time.sleep(0.5)
        planner.close_gripper()
        time.sleep(0.5)
        # sub_stage3: lift
        lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
        planner.move_to_pose_with_screw(lift_pose)
        is_grasp = env.evaluate()["is_cubeA_grasped"]
        time.sleep(0.5)
        # sub_stage4: stack
        goal_pose = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 4])
        # goal_pose, _, error_description, speed_factor = add_perturbation("stack", goal_pose)
        # Align the stack pose
        offset = (goal_pose.p - env.cubeA.pose.p).numpy()[0]  
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        planner.move_to_pose_with_screw(align_pose)
        time.sleep(0.5)

        res = planner.open_gripper()
        time.sleep(1)

        success_level = "critical failure"
        logs.append("In front of the robot arm, there were a peach, an apple, an orange, and a golf ball. The task was to grasp the peach, but the robot arm chose to grasp the apple, leading to the task's final failure.")
        error_description["stage"] = "grasp"
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

### 修改位置（行数）：85,91; 172,178; 257,263; 340,346; 425,431

def solve_stackcube_appleplate(env: StackCubeEnv_appleplate, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.cubeA)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    delta_angle = 0 
    delta_quat = euler2quat(0, 0, delta_angle)
    delta_pose = sapien.Pose(q=delta_quat)
    grasp_pose_offset = grasp_pose * delta_pose

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        # grasp_pose2 = grasp_pose * delta_pose
        grasp_pose2 = grasp_pose_offset * delta_pose # Add
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose 
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 4])  
    offset = (goal_pose.p - env.cubeA.pose.p).numpy()[0] 
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()
    time.sleep(0.5)
    planner.close()
    return res


if __name__ == '__main__':
    # quickstart
    env = gym.make(
        "StackCube-appleplate-more", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    obs, _ = env.reset(seed=0) # reset with a seed for determinism
    done = False
    tol_res = 0
    tol_count = 0
    log_file = 'log_test4'
    for seed in range(10):
        res = solve_with_errors(env, log_file=log_file,error_stage='none',perturbation_type="position_offset", seed=seed,vis=True)
        # res = solve_stackcube_appleplate(env,vis=True)
        print(res)
    #   break
    # seed = 5
    # p = 0
    # while(1):
    #     solve_with_errors(env,seed=seed,vis=True)
    #     p+=1
    #     if p == 10:
    #         break
    # print(tol_res,tol_count)
    env.close()
    