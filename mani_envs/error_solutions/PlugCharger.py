import gymnasium as gym
import numpy as np
import sapien.core as sapien
import trimesh
from tqdm import tqdm
from transforms3d.euler import euler2quat
import json
import time
import uuid
from mani_skill.envs.tasks import PlugChargerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

ERROR_DESCRIPTIONS = {
    "reach_failure": "The robot arm failed to reach above the plug, causing a grasp failure.",
    "reach_bad_position": "The robot arm did not align well above the plug, causing a bad grasp position and ultimately failing to insert the plug.",
    "reach_partial_success": "The robot arm did not align well above the plug, causing a bad grasp position but managed to complete the task with difficulty.",

    "grasp_position_failure": "The robot arm failed to grasp the plug due to a position offset. The misalignment caused the gripper to miss the plug entirely, resulting in a task failure.",
    "grasp_position_bad": "The robot arm grasped the plug but not in a good position, ultimately failing to insert the plug.",
    "grasp_position_partial_success": "The robot arm grasped the plug in a suboptimal position but managed to complete the task with difficulty.",
    
    "grasp_rotation_failure": "The robot arm failed to grasp the plug due to an incorrect orientation. The gripper's angle was misaligned, causing it to slip off the plug and resulting in a task failure.",
    "grasp_rotation_bad": "The robot arm grasped the plug but not in a good orientation, ultimately failing to insert the plug.",
    "grasp_rotation_partial_success": "The robot arm grasped the plug in a suboptimal orientation but managed to complete the task with difficulty.",
    
    "align_position_failure": "The robot arm failed to align the plug with the socket due to a position offset. The misalignment caused the plug to be positioned incorrectly relative to the socket, resulting in an insertion failure.",
    "align_position_partial_success": "The robot arm did not align the plug perfectly with the socket, causing some difficulty during insertion, but managed to insert it successfully with minor adjustments.",
    
    "align_rotation_failure": "The robot arm failed to align the plug with the socket due to a rotation offset. The incorrect orientation of the plug caused it to be misaligned with the socket, leading to an insertion failure.",
    "align_rotation_partial_success": "The robot arm did not align the plug perfectly with the socket due to a slight rotation offset, but managed to insert it successfully with some difficulty.",
    
    "insert_failure": "The plug was initially aligned with the socket but deviated during insertion. The deviation caused the plug to miss the socket, resulting in a task failure.",
    "insert_partial_success": "The plug was aligned with the socket and experienced slight deviation during insertion, causing some difficulty, but managed to complete the task with minor adjustments."
}
def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

# Add perturbation to specific subtasks
def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type=None):
    error_description = {}
    if task_stage == "reach":
        perturbation = np.zeros(3)
        for i in range(3):
            if np.random.rand() > 0.5:
                    perturbation[i] = np.random.uniform(-0.08, -0.05)
            else:
                    perturbation[i] = np.random.uniform(0.05, 0.08)

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
        error_description["root_cause"] = "The robot arm failed to reach above the plug."
        error_description["correction_suggestion"] = "Adjust the robot arm's trajectory to ensure it reaches the correct position above the plug before attempting to grasp it."
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  # 方向向量
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
            error_description["error_type"] = "position offset"
            error_description["root_cause"] = "The robot arm failed to grasp the plug due to a position offset."
            error_description["correction_suggestion"] = "Adjust the robot arm's trajectory to ensure it reaches the correct grasping position before attempting to grasp it."

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
            error_description["original_pose"] = pose.tolist()
            pose = (pose * sapien.Pose(q=rotation_quat))
            # grasp_angle = np.deg2rad(15)
            # grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0)) 
            # rotation_quat = euler2quat(rotation_angle, rotation_angle, 0)  # Apply rotation around Y-axis
            # pose = (pose * sapien.Pose(q=rotation_quat))
            error_description["stage"] = "grasp"
            error_description["error_type"] = "rotation offset"
            error_description["root_cause"] = "The robot arm failed to grasp the plug due to a rotation offset."
            error_description["correction_suggestion"] = "Adjust the robot arm's trajectory to ensure the gripper is correctly oriented before attempting to grasp the plug."
 
            error_description["perturbation"] = rotation_angle.tolist()  
            error_description["perturbed_pose"] = pose.q.tolist()  
            error_description["direction_vector"] = rotation_quat.tolist()  

            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "align":
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

            # pose.p[:2] += alignment_offset[:2]  # Misalignment in X-Y plane
            error_description["stage"] = "align"
            error_description["error_type"] = "position offset"
            error_description["root_cause"] = "The robot arm failed to align the plug with the socket due to a position offset."
            error_description["correction_suggestion"] = "Adjust the robot arm's trajectory to ensure the plug is correctly aligned with the socket before attempting to insert it."
           
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
            error_description["original_pose"] = pose.tolist()
            pose = (pose * sapien.Pose(q=rotation_quat))

            error_description["stage"] = "grasp"
            error_description["error_type"] = "rotation offset"
            error_description["root_cause"] = "The robot arm failed to align the plug with the socket due to a rotation offset."
            error_description["correction_suggestion"] = "Adjust the robot arm's trajectory to ensure the plug is correctly oriented before attempting to insert it."

            directions = ["roll", "pitch", "yaw"]
            direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, rotation_angle)]
            error_description["direction_description"] = ", ".join(direction_desc)

    elif task_stage == "insert":
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
        error_description["stage"] = "insert"
        error_description["error_type"] = "position offset"
        error_description["root_cause"] = "The robot arm did not select the correct target position, resulting in a failure to insert into the socket."
        error_description["correction_suggestion"] = "Adjust the robot arm's trajectory to ensure the plug is inserted into the socket correctly."
   
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
 
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
        
    return pose, gripper_state, error_description

def main():
    env: PlugChargerEnv = gym.make(
        "PlugCharger-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
    )
    for seed in tqdm(range(100)):
        # res = solve(env, seed=seed, debug=False, vis=True)
        res = solve_with_errors(env, "plug_charger_log", error_stage="insert", seed=seed, debug=False, vis=True)
        print(res)
        time.sleep(1)

    env.close()

def solve_with_errors(env: PlugChargerEnv, log_file, error_stage, perturbation_type="position_offset", seed=None, debug=False, vis=False):
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
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
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

    ##########################################

    if error_stage == "reach":
        # add a angle to grasp
        grasp_angle = np.deg2rad(15)
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))
        time.sleep(0.5)
        # Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05]) 
        reach_pose,_,error_description = add_perturbation(error_stage, reach_pose)
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # Grasp
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.charger)
        time.sleep(0.5)
        # Align
        pre_insert_pose = (
            env.goal_pose.sp
            * sapien.Pose([-0.05, 0.0, 0.0])
            * env.charger.pose.sp.inv()
            * env.agent.tcp.pose.sp
        )
        insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0) 
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5) 
        time.sleep(0.5)
        # Insert
        res = planner.move_to_pose_with_screw(insert_pose)
        result_summary = env.evaluate()

        if not is_grasped:
            logs.append(ERROR_DESCRIPTIONS["reach_failure"])
            success_level = "critical failure"
        elif is_grasped and not result_summary["success"]:
            logs.append(ERROR_DESCRIPTIONS["reach_bad_position"])
            success_level = "partial success"
        else:
            logs.append(ERROR_DESCRIPTIONS["reach_partial_success"])
            success_level = "partial success"
        print(logs)
    elif error_stage == "grasp":
        # add a angle to grasp
        grasp_angle = np.deg2rad(15)
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0)) 
        time.sleep(0.5)
        # Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05]) 
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # Grasp
        # perturbation_type = np.random.choice(["position_offset", "rotation_offset"])
        # perturbation_type = "rotation_offset"
        grasp_pose,_,error_description = add_perturbation(error_stage, grasp_pose, perturbation_type=perturbation_type) # add perturbation
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.charger)
        time.sleep(0.5)
        # Align
        pre_insert_pose = (
            env.goal_pose.sp
            * sapien.Pose([-0.05, 0.0, 0.0])
            * env.charger.pose.sp.inv()
            * env.agent.tcp.pose.sp
        )
        insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0) 
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5) 
        time.sleep(0.5)
        # Insert
        res = planner.move_to_pose_with_screw(insert_pose)
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
                logs.append(ERROR_DESCRIPTIONS["grasp_rotation_bad"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["grasp_rotation_partial_success"])
                success_level = "partial success"
        print(logs)
    elif error_stage == "align":
        # add a angle to grasp
        grasp_angle = np.deg2rad(15)
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0)) 
        time.sleep(0.5)
        # Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05]) 
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # Grasp
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        is_grasped = env.agent.is_grasping(env.charger)
        time.sleep(0.5)
        # Align
        pre_insert_pose = (
            env.goal_pose.sp
            * sapien.Pose([-0.05, 0.0, 0.0])
            * env.charger.pose.sp.inv()
            * env.agent.tcp.pose.sp
        )
        insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
        # perturbation_type = "position_offset"
        pre_insert_pose,_,error_description = add_perturbation(error_stage, pre_insert_pose, perturbation_type=perturbation_type) # add perturbation
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0) 
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5) 
        time.sleep(0.5)
        # Insert
        res = planner.move_to_pose_with_screw(insert_pose)
        result_summary = env.evaluate()
        
        if perturbation_type == "position_offset":
            if not result_summary["success"]:
                logs.append(ERROR_DESCRIPTIONS["align_position_failure"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["align_position_partial_success"])
                success_level = "partial success"
        elif perturbation_type == "rotation_offset":
            if not result_summary["success"]:
                logs.append(ERROR_DESCRIPTIONS["align_rotation_failure"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["align_rotation_partial_success"])
                success_level = "partial success"

        print(logs)
    elif error_stage == "insert":
        # add a angle to grasp
        grasp_angle = np.deg2rad(15)
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0)) 
        time.sleep(0.5)
        # Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05]) 
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        # Grasp
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        time.sleep(0.5)
        # Align
        pre_insert_pose = (
            env.goal_pose.sp
            * sapien.Pose([-0.05, 0.0, 0.0])
            * env.charger.pose.sp.inv()
            * env.agent.tcp.pose.sp
        )
        insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0) 
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5) 
        time.sleep(0.5)
        # Insert
        insert_pose,_,error_description = add_perturbation(error_stage, insert_pose)
        res = planner.move_to_pose_with_screw(insert_pose)
        result_summary = env.evaluate()
        if not result_summary["success"]:
            logs.append(ERROR_DESCRIPTIONS["insert_failure"])
            success_level = "critical failure"
        else:
            logs.append(ERROR_DESCRIPTIONS["insert_partial_success"])
            success_level = "partial success"
        print(logs)    

    planner.close()
   
    simulation_id = str(uuid.uuid4())
    error_description["details"] = logs[0]
    log_entry = {
        "simulation_id": simulation_id,
        "success_level": success_level,
        # "details": logs,
        "error_description":error_description
    }

    with open(log_file, 'a') as f:
        json.dump(log_entry, f, indent=4)
        f.write('\n')
    return res, simulation_id

def solve(env: PlugChargerEnv, seed=None, debug=False, vis=False):
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
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
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
    time.sleep(0.5)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    time.sleep(0.5)


    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    time.sleep(0.5)


    # -------------------------------------------------------------------------- #
    # Align
    # -------------------------------------------------------------------------- #
    pre_insert_pose = (
        env.goal_pose.sp
        * sapien.Pose([-0.05, 0.0, 0.0])
        * env.charger.pose.sp.inv()
        * env.agent.tcp.pose.sp
    )
    insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
    planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0) 
    planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5) 
    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(insert_pose)

    planner.close()
    return res


if __name__ == "__main__":
    main()
