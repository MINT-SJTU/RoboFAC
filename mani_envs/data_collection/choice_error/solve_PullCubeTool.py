import numpy as np
import sapien

from mani_skill.envs.tasks import PullCubeToolEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from choice_Error.task_PullCubeTool import PullCubeToolEnv_golf_more, PullCubeToolEnv_dice_more, PullCubeToolEnv_can_more, PullCubeToolEnv_peach_more, PullCubeToolEnv_marble_more

def solve_pullcubetool_golf_more(env: PullCubeToolEnv_golf_more, seed=None, debug=False, vis=False):
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
        [-(env.hook_length + 4 * env.cube_half_size),   # adjusted position
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


def solve_pullcubetool_dice_more(env: PullCubeToolEnv_dice_more, seed=None, debug=False, vis=False):
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
        [-(env.hook_length + 4 * env.cube_half_size),   # adjusted position
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

def solve_pullcubetool_can_more(env: PullCubeToolEnv_can_more, seed=None, debug=False, vis=False):
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
        [-(env.hook_length + 4 * env.cube_half_size),  # adjusted position 
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

def solve_pullcubetool_peach_more(env: PullCubeToolEnv_peach_more, seed=None, debug=False, vis=False):
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
        [-(env.hook_length + 4 * env.cube_half_size),   # adjusted position
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

def solve_pullcubetool_marble_more(env: PullCubeToolEnv_marble_more, seed=None, debug=False, vis=False):
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
        [-(env.hook_length + 4 * env.cube_half_size),   # adjusted position
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