#!/usr/bin/env python3
import subprocess
import time
import argparse
import torch

def run_stage(env: str, stage: str, view: int, perturbation_type: str):

    error_stage = stage
    stage_dir = f"{error_stage}_view{view}_{perturbation_type}"  

    # Construct the command argument list
    cmd = [
        "python",
        "/home/idphilosea/miniconda3/envs/maniskill/lib/python3.10/site-packages/mani_skill/examples/motionplanning/panda/generalization/generalization/collect.py",
        "-e", env,
        "-n", "10",
        "--save-video",
        "--traj-name", "trajctory",
        "--record-dir", "error_collection",
        "--num-procs", "1",
        "--error-stage", error_stage,
        "--perturbation-type", perturbation_type,  
        "--json-file", "error_description.json",
        "--stage-dir", stage_dir
    ]
    
    print(f"Starting error-stage: {error_stage} with perturbation-type: {perturbation_type}")
    print("Command: " + " ".join(cmd))
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error occurred in error-stage {error_stage} with perturbation-type {perturbation_type}!")
    else:
        print(f"Error-stage {error_stage} with perturbation-type {perturbation_type} completed successfully.")

if __name__ == '__main__':
    stack_envs = ["StackCube-v1", "StackCube-appleplate", "StackCube-can", "StackCube-lego", "StackCube-orangebowl", "StackCube-dicebrick"]
    
    tool_envs = ["PullCubeTool-v1", "PullCubeTool-golf", "PullCubeTool-dice", "PullCubeTool-can", "PullCubeTool-peach", "PullCubeTool-marble" ]

    pull_envs = ["PullCube-v1","PullCube-lego","PullCube-scissor","PullCube-block"]

    push_envs = ["PushCube-v1","PushCube-box","PushCube-cup","PushCube-toy"]

    plug_envs = {"PlugCharger-v1"}
    # Define the error-stage list and their corresponding perturbation types to collect
    error_stages_stack = {
        "reach": ["position_offset", "rotation_offset"],
        "grasp": ["position_offset", "rotation_offset"],
        "lift": ["position_offset"],
        "stack": ["position_offset"],
    }
    error_stages_tool = {
        "reach": ["position_offset", "rotation_offset"],
        "grasp": ["position_offset", "rotation_offset"],
        "lift_tool": ["speed_offset"],
        "lower_tool": ["position_offset"],
        "pull": ["position_offset"],
    }
    error_stages_pull = {
        "reach": ["position_offset"],
        "close_gripper": ["close_late"],
        "pull": ["position_offset"],
    }
    error_stages_push = {
        "reach": ["position_offset"],
        "close_gripper": ["close_late"],
        "push": ["position_offset"],
    }
    error_stages_plug = {
        "reach": ["position_offset"],
        "grasp": ["position_offset", "rotation_offset"],
        "align": ["position_offset", "rotation_offset"],
        "insert": ["position_offset"],
    }
    
    error_stages = error_stages_tool # Select the error-stage to collect
    envs = tool_envs # Select the environments to collect

    for env in envs:
        for stage, perturbation_types in error_stages.items():
            for perturbation_type in perturbation_types:
                run_stage(env, stage, view=0, perturbation_type=perturbation_type)
                # Delay to ensure resources are fully released
                print("Waiting 3 seconds to release resources...\n")
                torch.cuda.empty_cache()
                time.sleep(3)