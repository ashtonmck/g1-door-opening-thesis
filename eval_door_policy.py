"""
Evaluate trained GR00T policy on door-opening task in MuJoCo.

Loads the finetuned checkpoint, runs N episodes in the MuJoCo door scene,
and measures success rate (door hinge angle > threshold).

Usage:
    conda activate unitree_sim_env
    cd ~/groot
    python ~/eval_door_policy.py \
        --checkpoint ~/groot_checkpoint \
        --scene ~/unitree_mujoco/unitree_robots/g1/scene_29dof_door.xml \
        --n-episodes 20 \
        --max-steps 300 \
        --render
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer
import mujoco
from mujoco import viewer as mj_viewer
import cv2

# Add groot to path
sys.path.insert(0, os.path.expanduser('~/groot'))


def get_arm_addresses(mj_model):
    """Pre-compute arm joint qpos/qvel addresses."""
    LEFT_ARM_START = 15
    RIGHT_ARM_START = 22

    arm_qpos_adr = []
    arm_qvel_adr = []
    for i in range(7):
        jnt_id_l = mj_model.actuator_trnid[LEFT_ARM_START + i, 0]
        arm_qpos_adr.append(mj_model.jnt_qposadr[jnt_id_l])
        arm_qvel_adr.append(mj_model.jnt_dofadr[jnt_id_l])
        jnt_id_r = mj_model.actuator_trnid[RIGHT_ARM_START + i, 0]
        arm_qpos_adr.append(mj_model.jnt_qposadr[jnt_id_r])
        arm_qvel_adr.append(mj_model.jnt_dofadr[jnt_id_r])

    leg_qpos_adr = []
    leg_qvel_adr = []
    for i in range(15):
        jnt_id = mj_model.actuator_trnid[i, 0]
        leg_qpos_adr.append(mj_model.jnt_qposadr[jnt_id])
        leg_qvel_adr.append(mj_model.jnt_dofadr[jnt_id])

    return arm_qpos_adr, arm_qvel_adr, leg_qpos_adr, leg_qvel_adr


def get_arm_state(mj_data, arm_qpos_adr):
    """Extract current arm joint positions as [left_7, right_7]."""
    left_arm = np.array([mj_data.qpos[arm_qpos_adr[i * 2]] for i in range(7)])
    right_arm = np.array([mj_data.qpos[arm_qpos_adr[i * 2 + 1]] for i in range(7)])
    return left_arm, right_arm


def apply_arm_action(mj_data, action_left, action_right, arm_qpos_adr, arm_qvel_adr,
                     kp=80.0, kd=6.0):
    """Apply PD control to arm joints."""
    LEFT_ARM_START = 15
    RIGHT_ARM_START = 22

    for i in range(7):
        # Left arm
        idx_l = LEFT_ARM_START + i
        cur_pos_l = mj_data.qpos[arm_qpos_adr[i * 2]]
        cur_vel_l = mj_data.qvel[arm_qvel_adr[i * 2]]
        mj_data.ctrl[idx_l] = kp * (action_left[i] - cur_pos_l) - kd * cur_vel_l

        # Right arm
        idx_r = RIGHT_ARM_START + i
        cur_pos_r = mj_data.qpos[arm_qpos_adr[i * 2 + 1]]
        cur_vel_r = mj_data.qvel[arm_qvel_adr[i * 2 + 1]]
        mj_data.ctrl[idx_r] = kp * (action_right[i] - cur_pos_r) - kd * cur_vel_r


def render_camera(renderer, mj_data, cam_id, width=640, height=480):
    """Render camera view and return RGB uint8 array."""
    renderer.update_scene(mj_data, camera=cam_id)
    img = renderer.render()
    return img  # RGB uint8


def get_door_angle(mj_model, mj_data):
    """Get the door hinge angle in radians."""
    door_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'door_hinge')
    if door_jnt_id < 0:
        return 0.0
    qpos_adr = mj_model.jnt_qposadr[door_jnt_id]
    return mj_data.qpos[qpos_adr]


def run_episode(policy, mj_model, mj_data, renderer, cam_id,
                arm_qpos_adr, arm_qvel_adr, leg_qpos_adr, leg_qvel_adr,
                initial_qpos, max_steps=300, render=False, viewer=None,
                action_horizon=16, n_action_steps=8):
    """Run one episode of the policy in MuJoCo."""

    LEG_KP = 200.0
    LEG_KD = 10.0

    # Reset
    mj_data.qpos[:] = initial_qpos
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    policy.reset()

    door_angles = []
    step = 0
    action_buffer = None
    action_buffer_idx = 0

    while step < max_steps:
        # Get observation for policy (every n_action_steps)
        if action_buffer is None or action_buffer_idx >= n_action_steps:
            # Render camera image
            img = render_camera(renderer, mj_data, cam_id)  # (H, W, 3) RGB uint8

            # Get arm state
            left_arm, right_arm = get_arm_state(mj_data, arm_qpos_adr)

            # Build observation in GR00T format
            # State: [left_arm(7), right_arm(7), left_ee(7), right_ee(7)] = 28
            state = np.concatenate([
                left_arm,           # left_arm (7)
                right_arm,          # right_arm (7)
                np.zeros(7),        # left_ee placeholder (Dex3 not in eval)
                np.zeros(7),        # right_ee placeholder
            ]).astype(np.float32)

            # Create blank wrist images (policy expects 3 cameras)
            blank_img = np.zeros((480, 640, 3), dtype=np.uint8)

            observation = {
                "video": {
                    "head": img[np.newaxis, np.newaxis, ...],           # (1, 1, H, W, 3)
                    "left_wrist": blank_img[np.newaxis, np.newaxis, ...],
                    "right_wrist": blank_img[np.newaxis, np.newaxis, ...],
                },
                "state": {
                    "left_arm": left_arm[np.newaxis, np.newaxis, ...].astype(np.float32),    # (1, 1, 7)
                    "right_arm": right_arm[np.newaxis, np.newaxis, ...].astype(np.float32),
                    "left_ee": np.zeros((1, 1, 7), dtype=np.float32),
                    "right_ee": np.zeros((1, 1, 7), dtype=np.float32),
                },
                "language": {
                    "annotation.human.action.task_description": [["push the door open"]],
                },
            }

            # Get action from policy
            action, info = policy.get_action(observation)

            # Extract action chunks
            # action keys match modality config: left_arm, right_arm, left_ee, right_ee
            if "left_arm" in action:
                left_arm_actions = action["left_arm"][0]    # (T, 7)
                right_arm_actions = action["right_arm"][0]  # (T, 7)
            else:
                # Fallback: single action key with all 28 dims
                all_actions = list(action.values())[0][0]  # (T, 28)
                left_arm_actions = all_actions[:, :7]
                right_arm_actions = all_actions[:, 7:14]

            action_buffer = (left_arm_actions, right_arm_actions)
            action_buffer_idx = 0

        # Apply current action from buffer
        left_target = action_buffer[0][min(action_buffer_idx, len(action_buffer[0]) - 1)]
        right_target = action_buffer[1][min(action_buffer_idx, len(action_buffer[1]) - 1)]
        action_buffer_idx += 1

        # PD control for arms
        apply_arm_action(mj_data, left_target, right_target,
                         arm_qpos_adr, arm_qvel_adr)

        # Fix floating base
        mj_data.qpos[:3] = initial_qpos[:3]
        mj_data.qpos[3:7] = initial_qpos[3:7]
        mj_data.qvel[:6] = 0.0

        # Leg PD
        for i in range(15):
            mj_data.ctrl[i] = LEG_KP * (initial_qpos[leg_qpos_adr[i]] - mj_data.qpos[leg_qpos_adr[i]]) \
                            - LEG_KD * mj_data.qvel[leg_qvel_adr[i]]

        # Step simulation (multiple substeps per control step)
        for _ in range(10):
            mujoco.mj_step(mj_model, mj_data)

        # Record door angle
        angle = get_door_angle(mj_model, mj_data)
        door_angles.append(angle)

        # Render viewer
        if render and viewer is not None and viewer.is_running():
            viewer.sync()

        step += 1

    max_angle = max(door_angles) if door_angles else 0.0
    final_angle = door_angles[-1] if door_angles else 0.0
    return max_angle, final_angle, door_angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=os.path.expanduser('~/groot_checkpoint'),
                        help='Path to GR00T checkpoint directory')
    parser.add_argument('--scene', type=str,
                        default=os.path.expanduser('~/unitree_mujoco/unitree_robots/g1/scene_29dof_door.xml'),
                        help='Path to MuJoCo scene XML')
    parser.add_argument('--n-episodes', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Max control steps per episode')
    parser.add_argument('--success-threshold', type=float, default=0.3,
                        help='Door angle threshold (radians) for success (~17 degrees)')
    parser.add_argument('--render', action='store_true', help='Show MuJoCo viewer')
    parser.add_argument('--save-video', action='store_true', help='Save episode videos')
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~/eval_results'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MuJoCo scene
    print(f"Loading scene: {args.scene}")
    mj_model = mujoco.MjModel.from_xml_path(args.scene)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = 0.001

    # Get joint addresses
    arm_qpos_adr, arm_qvel_adr, leg_qpos_adr, leg_qvel_adr = get_arm_addresses(mj_model)
    initial_qpos = mj_data.qpos.copy()

    # Camera setup
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if cam_id < 0:
        print("WARNING: No head_camera found in scene")

    # Viewer
    viewer = None
    if args.render:
        viewer = mj_viewer.launch_passive(mj_model, mj_data)

    # Load GR00T policy
    print(f"Loading policy from: {args.checkpoint}")
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device="cuda:0",
        strict=False,
    )

    print(f"\n{'='*60}")
    print(f"Running {args.n_episodes} evaluation episodes")
    print(f"Success threshold: {args.success_threshold:.2f} rad ({np.degrees(args.success_threshold):.1f}°)")
    print(f"{'='*60}\n")

    # Run evaluation
    results = []
    successes = 0

    for ep in range(args.n_episodes):
        max_angle, final_angle, angles = run_episode(
            policy, mj_model, mj_data, renderer, cam_id,
            arm_qpos_adr, arm_qvel_adr, leg_qpos_adr, leg_qvel_adr,
            initial_qpos, max_steps=args.max_steps,
            render=args.render, viewer=viewer,
        )

        success = max_angle > args.success_threshold
        if success:
            successes += 1

        status = "✅ SUCCESS" if success else "❌ FAIL"
        print(f"Episode {ep+1:3d}/{args.n_episodes}: max_angle={np.degrees(max_angle):6.1f}° "
              f"final_angle={np.degrees(final_angle):6.1f}° {status}")

        results.append({
            "episode": ep,
            "max_angle_rad": max_angle,
            "max_angle_deg": np.degrees(max_angle),
            "final_angle_rad": final_angle,
            "final_angle_deg": np.degrees(final_angle),
            "success": success,
        })

        # Save angle trajectory
        if args.save_video:
            np.save(os.path.join(args.output_dir, f"angles_ep{ep:03d}.npy"), angles)

    # Print summary
    success_rate = successes / args.n_episodes * 100
    avg_max_angle = np.mean([r["max_angle_deg"] for r in results])
    avg_final_angle = np.mean([r["final_angle_deg"] for r in results])

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes:        {args.n_episodes}")
    print(f"Success rate:    {successes}/{args.n_episodes} ({success_rate:.1f}%)")
    print(f"Avg max angle:   {avg_max_angle:.1f}°")
    print(f"Avg final angle: {avg_final_angle:.1f}°")
    print(f"Threshold:       {np.degrees(args.success_threshold):.1f}°")
    print(f"{'='*60}")

    # Save results
    import json
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "config": vars(args),
            "summary": {
                "n_episodes": args.n_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "avg_max_angle_deg": avg_max_angle,
                "avg_final_angle_deg": avg_final_angle,
            },
            "episodes": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
