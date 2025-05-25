import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube

from solutions.solve_PickCube import solve_pickcube_apple, solve_pickcube_can, solve_pickcube_ball, solve_pickcube_rubiks, solve_pickcube_lego, solve_pickcube_lock
from solutions.solve_StackCube import solve_stackcube_appleplate, solve_stackcube_can, solve_stackcube_lego, solve_stackcube_orangebowl, solve_stackcube_dicebrick
from solutions.solve_PushCube import solve_pushcube_box, solve_pushcube_cup, solve_pushcube_toy
from solutions.solve_PullCube import solve_pullcube_lego, solve_pullcube_scissor, solve_pullcube_block
from solutions.solve_PullCubeTool import solve_pullcubetool_golf, solve_pullcubetool_dice, solve_pullcubetool_can, solve_pullcubetool_peach, solve_pullcubetool_marble
from solutions.solve_LiftPegUpright import solve_liftpegupright_box, solve_liftpegupright_can, solve_liftpegupright_cup
# from task_generalization.solve_PegInsertionSide import solve_peginsertionside_screwdriv
from solutions.solve_Safe import solveSafe, solveSafe_hammer, solveSafe_usb, solveSafe_spatula, solveSafe_screwdriver
from solutions.solve_Microwave import solveMicrowave, solveMicrowave_fork, solveMicrowave_knife, solveMicrowave_mug
from solutions.solve_Tools import solveTools
from solutions.solve_UprightStack import solveUprightStack, solveUprightStack_gen1, solveUprightStack_gen2

from solutions.solve_SpinStack import solveRotation_gen1, solveRotation_gen2, solveRotation
from solutions.solve_SpinPullStack import solveRotation2, solveRotation2_gen1, solveRotation2_gen2

from solutions import *
CORRECT_SOLUTIONS = {
    "PickCube-v1": solvePickCube,
    "PickCube-apple": solve_pickcube_apple,
    "PickCube-can": solve_pickcube_can,
    "PickCube-ball": solve_pickcube_ball,
    "PickCube-rubiks": solve_pickcube_rubiks,
    "PickCube-lego": solve_pickcube_lego,
    "PickCube-lock": solve_pickcube_lock,
    "StackCube-v1": solveStackCube,
    "StackCube-appleplate": solve_stackcube_appleplate,
    "StackCube-can": solve_stackcube_can,
    "StackCube-lego": solve_stackcube_lego,
    "StackCube-orangebowl": solve_stackcube_orangebowl,
    "StackCube-dicebrick": solve_stackcube_dicebrick,
    "PegInsertionSide-v1": solvePegInsertionSide,
    # "PegInsertionSide-screwdriver": solve_peginsertionside_screwdriver,
    "PlugCharger-v1": solvePlugCharger,
    "PushCube-v1": solvePushCube,
    "PushCube-box": solve_pushcube_box,
    "PushCube-cup": solve_pushcube_cup,
    "PushCube-toy": solve_pushcube_toy,
    "PullCubeTool-v1": solvePullCubeTool,
    "PullCubeTool-golf": solve_pullcubetool_golf,
    "PullCubeTool-dice": solve_pullcubetool_dice,
    "PullCubeTool-can": solve_pullcubetool_can,
    "PullCubeTool-peach": solve_pullcubetool_peach,
    "PullCubeTool-marble": solve_pullcubetool_marble,
    "LiftPegUpright-v1": solveLiftPegUpright,
    "LiftPegUpright-box": solve_liftpegupright_box,
    "LiftPegUpright-can": solve_liftpegupright_can,
    "LiftPegUpright-cup": solve_liftpegupright_cup,
    "PullCube-v1": solvePullCube,
    "PullCube-lego": solve_pullcube_lego,
    "PullCube-scissor": solve_pullcube_scissor,
    "PullCube-block": solve_pullcube_block,
    "MicrowaveTask-v1": solveMicrowave,
    "MicrowaveTask-fork": solveMicrowave_fork,
    "MicrowaveTask-knife": solveMicrowave_knife,
    "MicrowaveTask-mug": solveMicrowave_mug,
    "SafeTask": solveSafe,
    "SafeTask-usb": solveSafe_usb,
    "SafeTask-hammer": solveSafe_hammer,
    "SafeTask-screwdriver": solveSafe_screwdriver,
    "SafeTask-spatula": solveSafe_spatula,
    "UprightStack-v1": solveUprightStack,
    "UprightStack-gen1": solveUprightStack_gen1,
    "UprightStack-gen2": solveUprightStack_gen2,
    "ToolsTask-v1": solveTools,
    "SpinStack-v1": solveRotation,
    "SpinStack-gen1": solveRotation_gen1,
    "SpinStack-gen2": solveRotation_gen2,
    "SpinPullStack-v1": solveRotation2,
    "SpinPullStack-gen1": solveRotation2_gen1,
    "SpinPullStack-gen2": solveRotation2_gen2,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="StackCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(CORRECT_SOLUTIONS.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=5, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, default="StackCube", help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories") # change the record-dir
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    # add error-stage and perturbation-type
    parser.add_argument("--error-stage", type=str, default="stack", help="Stage at which to introduce errors.")
    parser.add_argument("--perturbation-type", type=str, default="position_offset", help="Type of perturbation to introduce.")
    parser.add_argument("--json-file", type=str, default="StackCube_stack_error.json", help="File to save the error logs.")
    parser.add_argument("--stage-dir", type=str, default="stack_error", help="store the stage of the error")
    return parser.parse_args()

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="dense" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend
    )
    if env_id not in CORRECT_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(CORRECT_SOLUTIONS.keys())}")
    
    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, args.stage_dir), ## change the output_dir
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        save_on_reset=False
    )
    output_dir=str(osp.join(args.record_dir, env_id, args.stage_dir)), ## change the output_dir
    output_h5_path = env._h5_file.filename
    # solve_error = MP_SOLUTIONS[env_id]
    solve = CORRECT_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    while True:
        try:
            res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
        except Exception as e:
            print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths),
                    max_episode_length=np.max(solution_episode_lengths),
                    # min_episode_length=np.min(solution_episode_lengths)
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break
    env.close()
    return output_h5_path

def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())