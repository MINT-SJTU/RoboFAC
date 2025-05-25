import numpy as np
import sapien
import gymnasium as gym
import time
import uuid

from mani_skill.envs.tasks import PullCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.task_PullCube import PullCubeEnv_lego, PullCubeEnv_scissor, PullCubeEnv_block


ERROR_DESCRIPTIONS_TEMPLATE = {
    "reach_partial_success": "The gripper reached a suboptimal position but managed to pull the {object} close to the target.",
    "reach_failure": "The gripper reached a bad position, causing the {object} to slip during the pull and fail to reach the target.",
    "reach_collision": "The gripper collided with the {object} during the reach phase, causing the {object} to slip and fail to reach the target.",
    "grasp_failure": "The gripper closed too late, failing to grasp the {object} and pull it to the target.",
    "pull_partial_success": "The gripper pulled the {object} but it deviated slightly from the target position.",
    "pull_failure": "The gripper failed to pull the {object} to the target position due to significant deviation."
}

def get_error_descriptions(object):
    return {key: value.format(object=object) for key, value in ERROR_DESCRIPTIONS_TEMPLATE.items()}

ERROR_DESCRIPTIONS = get_error_descriptions(object="lego")

def check_collision(env, entity1, entity2):

    contacts = env.scene.get_contacts()  
    for contact in contacts:
        try:
            body1, body2 = contact.bodies
            if (body1 == entity1 and body2 == entity2) or (body1 == entity2 and body2 == entity1):
                return True
        except AttributeError as e:
            print(f"Error accessing contact bodies: {e}")
    return False


def record_error_log(log_file, error_data):
    with open(log_file, "a") as f:
        json.dump(error_data, f, indent=4)
        f.write("\n")

# Add perturbation to specific subtasks
def add_perturbation(task_stage, pose, gripper_state=None, perturbation_type=None):
    error_description = {}
    object = "lego"
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
        error_description["root_cause"] = "The gripper reached a bad position."
        error_description["correction_suggestion"] = f"Adjust the reach position to align with the {object}."
        # error_description["details"] = f"Position offset by {perturbation.tolist()}."
       
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
     

    elif task_stage == "close_gripper":
        if perturbation_type == "early_close":
            gripper_state = "close_early"
            error_description["stage"] = "grasp"
            error_description["error_type"] = "gripper close early"
            # error_description["details"] = "Gripper closed before aligning with the {object}."
        elif perturbation_type == "late_close":
            gripper_state = "close_late"
            error_description["stage"] = "grasp"
            error_description["error_type"] = "gripper error"
            error_description["root_cause"] = f"The gripper closed after passing the {object}."
            error_description["correction_suggestion"] = f"Close the gripper before passing the {object}."
            # error_description["details"] = f"Gripper closed after passing the {object}."

    elif task_stage == "pull":
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
        error_description["root_cause"] = f"The gripper pulled the {object} in the wrong direction."
        error_description["correction_suggestion"] = "Adjust the pull direction to align with the target."
        # error_description["details"] = f"Direction offset by {direction_offset.tolist()}."
       
        error_description["perturbation"] = perturbation.tolist()
        error_description["perturbed_pose"] = pose.p.tolist()
        error_description["direction_vector"] = direction_vector.tolist()  
        
        directions = ["x", "y", "z"]
        direction_desc = [f"{d}{p:.3f}" for d, p in zip(directions, perturbation)]
        error_description["direction_description"] = ", ".join(direction_desc)
     

    return pose, gripper_state, error_description

def solve_with_errors(env: PullCubeEnv,log_file, error_stage, perturbation_type="position_offset",  seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    logs = []
    success_level = ""
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

    if error_stage == "close_gripper":
        perturbation_type = "late_close"
        _,_,error_description = add_perturbation(error_stage, sapien.Pose(), perturbation_type=perturbation_type)
        # Reach
        reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([0.08, 0, 0]), q=env.agent.tcp.pose.sp.q)
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([0.05, 0, 0]),q=env.agent.tcp.pose.sp.q)
        res = planner.move_to_pose_with_screw(goal_pose)
        planner.close_gripper()
        result = env.evaluate()
        if not result["success"]:
            logs.append(ERROR_DESCRIPTIONS["grasp_failure"])
            success_level = "critical failure"

    elif error_stage == "reach":
        planner.close_gripper() # close gripper
        time.sleep(0.5)
        reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([0.08, 0, 0]), q=env.agent.tcp.pose.sp.q)
        reach_pose,_,error_description = add_perturbation(error_stage, reach_pose) # Add perturbation
        planner.move_to_pose_with_screw(reach_pose) # Reach
        collision_flag = check_collision(env, env.obj, env.agent.robot)
        # print(collision_flag)
        time.sleep(0.5)
        goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([0.05, 0, 0]),q=env.agent.tcp.pose.sp.q)
        res = planner.move_to_pose_with_screw(goal_pose) # Move to goal pose
        result = env.evaluate()
        if result["success"]:
            logs.append(ERROR_DESCRIPTIONS["reach_partial_success"])
            success_level = "partial success"
        else:
            if collision_flag:
                logs.append(ERROR_DESCRIPTIONS["reach_collision"])
                success_level = "critical failure"
            else:
                logs.append(ERROR_DESCRIPTIONS["reach_failure"])
                success_level = "critical failure"
            # logs.append(ERROR_DESCRIPTIONS["reach_failure"])
        print(logs)

    elif error_stage == "pull":
        planner.close_gripper()
        time.sleep(0.5)
        reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([0.08, 0, 0]), q=env.agent.tcp.pose.sp.q)
        planner.move_to_pose_with_screw(reach_pose)
        time.sleep(0.5)
        goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([0.05, 0, 0]),q=env.agent.tcp.pose.sp.q)
        goal_pose,_,error_description = add_perturbation(error_stage, goal_pose)
        res = planner.move_to_pose_with_screw(goal_pose)
        result = env.evaluate()
        if result["success"]:
            logs.append(ERROR_DESCRIPTIONS["pull_partial_success"])
        else:
            logs.append(ERROR_DESCRIPTIONS["pull_failure"])

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

def solve_pullcube_lego(env: PullCubeEnv_lego, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([0.08, 0, 0]), q=env.agent.tcp.pose.sp.q) # 修改位置
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([0.05, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res

if __name__ == '__main__':
    # quickstart
    env = gym.make(
        "PullCube-lego", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    obs, _ = env.reset(seed=0) # reset with a seed for determinism
    done = False
    for seed in range(5):
            res = solve_with_errors(env,log_file='reach',error_stage='reach', seed=seed, debug=False, vis=True)
            time.sleep(1)
    print(res[-1])
    env.close()