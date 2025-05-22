import subprocess
import time
import torch

def run_stage(env: str, stage: str, view: int):
    cmd = [
        "python",
        "tasks/data_collect/run_correct.py",
        "-e", env,
        "-n", "10",
        "--save-video",
        "--traj-name", "trajctory",
        "--record-dir", "data_collection",
        "--num-procs", "1",
    ]
    
    print("Command: " + " ".join(cmd))
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"stage {stage} encountered an error!")
    else:
        print(f"stage {stage} completed successfully.")

if __name__ == '__main__':

    envs = [    
        "PickCube-v1",
        "PickCube-apple",
        "PickCube-can",
        "PickCube-ball",
        "PickCube-rubiks",
        "PickCube-lego",
        "PickCube-lock",
        "StackCube-v1",
        "StackCube-appleplate",
        "StackCube-lego",
        "StackCube-orangebowl",
        "StackCube-dicebrick",
        "PegInsertionSide-v1",
        "PlugCharger-v1",
        "PushCube-v1",
        "PushCube-box",
        "PushCube-cup",
        "PushCube-toy",
        "PullCubeTool-v1",
        "PullCubeTool-golf",
        "PullCubeTool-dice",
        "PullCubeTool-can",
        "PullCubeTool-peach",
        "PullCubeTool-marble",
        "LiftPegUpright-v1",
        "LiftPegUpright-box",
        "LiftPegUpright-can",
        "LiftPegUpright-cup",
        "PullCube-v1",
        "PullCube-lego",
        "PullCube-scissor",
        "PullCube-block",

        "MicrowaveTask-v1",
        "MicrowaveTask-fork",
        "MicrowaveTask-knife",
        "MicrowaveTask-mug",
        "ToolsTask-v1",
        "SafeTask",
        "SafeTask-usb",
        "SafeTask-hammer",
        "SafeTask-screwdriver",
        "SafeTask-spatula",
        "UprightStack-v1",
        "UprightStack-gen1",
        "UprightStack-gen2",
        "SpinStack-v1",
        "SpinStack-gen1",
        "SpinStack-gen2",
        "SpinPullStack-v1",
        "SpinPullStack-gen1",
        "SpinPullStack-gen2",
    ]      
    # Define the list of error stages to collect
    error_stages = [
        "none"
    ]
    for env in envs:
        for stage in error_stages:
            # run_stage(args.env, stage)
            run_stage(env, stage, view=0) # 1, 5, 3, 4 not 2
            # Delay to ensure resources are fully released
            print("Waiting 1 second to release resources...\n")
            torch.cuda.empty_cache()
            time.sleep(1)
