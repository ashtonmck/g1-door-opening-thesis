"""
Convert xr_teleoperate door_opening recordings to GR00T LeRobot v2 format.
"""
import json
import numpy as np
import pandas as pd
import cv2
import os
import subprocess
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path.home() / 'xr_teleoperate/teleop/utils/data/door_opening_mujoco'
OUTPUT_DIR  = Path.home() / 'groot_data/door_opening_mujoco'
TASK        = "open the door handle and push the door open"
FPS         = 30

# G1 29DOF with Dex3 joint names
LEFT_ARM_JOINTS  = [f"left_arm_{i}.pos"  for i in range(7)]
RIGHT_ARM_JOINTS = [f"right_arm_{i}.pos" for i in range(7)]
LEFT_EE_JOINTS   = [f"left_ee_{i}.pos"   for i in range(7)]
RIGHT_EE_JOINTS  = [f"right_ee_{i}.pos"  for i in range(7)]
ALL_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_EE_JOINTS + RIGHT_EE_JOINTS

# ── Setup output directories ──────────────────────────────────────────────────
(OUTPUT_DIR / 'data/chunk-000').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'videos/chunk-000/observation.images.head').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'videos/chunk-000/observation.images.left_wrist').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'videos/chunk-000/observation.images.right_wrist').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'meta').mkdir(parents=True, exist_ok=True)

# ── Find episodes ─────────────────────────────────────────────────────────────
episode_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"Found {len(episode_dirs)} episodes")

total_frames = 0
episodes_meta = []

for ep_idx, ep_dir in enumerate(episode_dirs):
    print(f"\nProcessing episode {ep_idx}: {ep_dir.name}")

    with open(ep_dir / 'data.json') as f:
        data = json.load(f)

    frames = data["data"]
    if len(frames) == 0:
        print(f"  Skipping - 0 frames")
        continue
    T = len(frames)
    print(f"  {T} frames")

    # ── Build state and action arrays ────────────────────────────────────────
    states  = []
    actions = []
    timestamps = []

    for t, frame in enumerate(frames):
        state = (
            frame['states']['left_arm']['qpos']  +
            frame['states']['right_arm']['qpos'] +
            frame['states']['left_ee']['qpos']   +
            frame['states']['right_ee']['qpos']
        )
        action = (
            frame['actions']['left_arm']['qpos']  +
            frame['actions']['right_arm']['qpos'] +
            frame['actions']['left_ee']['qpos']   +
            frame['actions']['right_ee']['qpos']
        )
        states.append(state)
        actions.append(action)
        timestamps.append(t / FPS)

    states  = np.array(states,  dtype=np.float32)   # (T, 28)
    actions = np.array(actions, dtype=np.float32)   # (T, 28)

    # ── Build parquet dataframe ──────────────────────────────────────────────
    df = pd.DataFrame({
        'observation.state': [s.tolist() for s in states],
        'action':            [a.tolist() for a in actions],
        'timestamp':         timestamps,
        'frame_index':       list(range(T)),
        'episode_index':     [ep_idx] * T,
        'index':             list(range(total_frames, total_frames + T)),
        'task_index':        [0] * T,
        'annotation.human.action.task_description': [0] * T,
        'annotation.human.validity': [1] * T,
        'next.reward':       [0] * T,
        'next.done':         [False] * (T-1) + [True],
    })

    parquet_path = OUTPUT_DIR / f'data/chunk-000/episode_{ep_idx:06d}.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved parquet: {parquet_path.name}")

    # ── Convert images to MP4 ───────────────────────────────────────────────
    # Load all frames for each camera
    for cam_idx, cam_name in enumerate(['head', 'left_wrist', 'right_wrist']):
        color_key = f'color_{cam_idx}'
        video_path = OUTPUT_DIR / f'videos/chunk-000/observation.images.{cam_name}/episode_{ep_idx:06d}.mp4'

        # Write frames to temp dir then encode
        tmp_dir = Path(f'/tmp/groot_ep{ep_idx}_{cam_name}')
        tmp_dir.mkdir(exist_ok=True)

        for t, frame in enumerate(frames):
            if color_key not in frame['colors']:
                continue
            img_path = ep_dir / frame['colors'][color_key]
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(tmp_dir / f'{t:06d}.jpg'), img)

        # Encode to MP4 using ffmpeg
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(FPS),
            '-i', str(tmp_dir / '%06d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f"  Saved video: {cam_name} ({video_path.name})")
        else:
            print(f"  WARNING: ffmpeg failed for {cam_name}: {result.stderr.decode()[:100]}")

        # Cleanup temp
        import shutil
        shutil.rmtree(tmp_dir)

    episodes_meta.append({
        "episode_index": ep_idx,
        "tasks": [{"task_index": 0, "task": TASK}],
        "length": T
    })
    total_frames += T
    print(f"  Episode {ep_idx} complete")

# ── Write meta files ──────────────────────────────────────────────────────────
print("\nWriting meta files...")

# tasks.jsonl
with open(OUTPUT_DIR / 'meta/tasks.jsonl', 'w') as f:
    f.write(json.dumps({"task_index": 0, "task": TASK}) + '\n')
    f.write(json.dumps({"task_index": 1, "task": "valid"}) + '\n')

# episodes.jsonl
with open(OUTPUT_DIR / 'meta/episodes.jsonl', 'w') as f:
    for ep in episodes_meta:
        f.write(json.dumps(ep) + '\n')

# modality.json - G1 29DOF with Dex3
modality = {
    "state": {
        "left_arm":  {"start": 0,  "end": 7},
        "right_arm": {"start": 7,  "end": 14},
        "left_ee":   {"start": 14, "end": 21},
        "right_ee":  {"start": 21, "end": 28},
    },
    "action": {
        "left_arm":  {"start": 0,  "end": 7},
        "right_arm": {"start": 7,  "end": 14},
        "left_ee":   {"start": 14, "end": 21},
        "right_ee":  {"start": 21, "end": 28},
    },
    "video": {
        "head":        {"original_key": "observation.images.head"},
        "left_wrist":  {"original_key": "observation.images.left_wrist"},
        "right_wrist": {"original_key": "observation.images.right_wrist"},
    },
    "annotation": {
        "human.action.task_description": {
            "original_key": "annotation.human.action.task_description"
        },
        "human.validity": {
            "original_key": "annotation.human.validity"
        }
    }
}
with open(OUTPUT_DIR / 'meta/modality.json', 'w') as f:
    json.dump(modality, f, indent=4)

# info.json
info = {
    "codebase_version": "v2.1",
    "robot_type": "unitree_g1_29dof_dex3",
    "total_episodes": len(episode_dirs),
    "total_frames": total_frames,
    "total_tasks": 1,
    "chunks_size": 1000,
    "fps": FPS,
    "splits": {"train": f"0:{len(episode_dirs)}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "names": ALL_JOINTS,
            "shape": [28]
        },
        "observation.state": {
            "dtype": "float32",
            "names": ALL_JOINTS,
            "shape": [28]
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480, "video.width": 640,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS, "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.images.left_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480, "video.width": 640,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS, "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480, "video.width": 640,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS, "video.channels": 3,
                "has_audio": False
            }
        },
    }
}
with open(OUTPUT_DIR / 'meta/info.json', 'w') as f:
    json.dump(info, f, indent=4)

print(f"\n✅ Conversion complete!")
print(f"   Episodes: {len(episode_dirs)}")
print(f"   Frames:   {total_frames}")
print(f"   Output:   {OUTPUT_DIR}")
