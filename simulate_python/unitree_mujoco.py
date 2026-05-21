import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import json
import base64

import zmq
import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from arm_shm_bridge import ArmShmReader

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


# --- Camera setup ---
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_ZMQ_PORT = 5555
CAM_PUBLISH_INTERVAL = 0.033  # ~30 fps


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)

    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    # --- Shared memory reader ---
    arm_reader = ArmShmReader(timeout_sec=0.5)

    # Arm actuator indices
    LEFT_ARM_START = 15
    RIGHT_ARM_START = 22

    # PD gains
    ARM_KP = 80.0
    ARM_KD = 6.0
    LEG_KP = 200.0
    LEG_KD = 10.0

    # Pre-compute arm joint addresses
    arm_qpos_adr = []
    arm_qvel_adr = []
    for i in range(7):
        jnt_id_l = mj_model.actuator_trnid[LEFT_ARM_START + i, 0]
        arm_qpos_adr.append(mj_model.jnt_qposadr[jnt_id_l])
        arm_qvel_adr.append(mj_model.jnt_dofadr[jnt_id_l])
        jnt_id_r = mj_model.actuator_trnid[RIGHT_ARM_START + i, 0]
        arm_qpos_adr.append(mj_model.jnt_qposadr[jnt_id_r])
        arm_qvel_adr.append(mj_model.jnt_dofadr[jnt_id_r])

    # Pre-compute leg + waist addresses
    leg_qpos_adr = []
    leg_qvel_adr = []
    for i in range(15):
        jnt_id = mj_model.actuator_trnid[i, 0]
        leg_qpos_adr.append(mj_model.jnt_qposadr[jnt_id])
        leg_qvel_adr.append(mj_model.jnt_dofadr[jnt_id])

    # Save full initial state for resets
    initial_qpos = mj_data.qpos.copy()
    initial_qpos[0] -= 0.0 # distance from door
    initial_qpos[1] -= 0.15
    initial_qvel = mj_data.qvel.copy()

    # Persistent arm target
    arm_target_q = None

    # --- Camera rendering setup ---
    renderer = mujoco.Renderer(mj_model, height=CAM_HEIGHT, width=CAM_WIDTH)
    head_cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    cam_enabled = head_cam_id >= 0
    if cam_enabled:
        print(f"[Camera] head_camera found (id={head_cam_id}), publishing on port {CAM_ZMQ_PORT}")
    else:
        print("[WARNING] No 'head_camera' found. Camera publishing disabled.")

    zmq_ctx = zmq.Context()
    zmq_pub = zmq_ctx.socket(zmq.PUB)
    zmq_pub.bind(f"tcp://*:{CAM_ZMQ_PORT}")

    last_cam_time = 0.0

    while viewer.is_running():
        try:
            step_start = time.perf_counter()

            locker.acquire()

            # --- Read arm commands ---
            sol_q, sol_tauff, is_new, is_valid, reset_requested = arm_reader.read()

            # --- Handle reset ---
            if reset_requested:
                print("[RESET] Resetting environment to initial state")
                mj_data.qpos[:] = initial_qpos
                mj_data.qvel[:] = initial_qvel
                mj_data.ctrl[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)
                arm_target_q = None
                locker.release()
                time.sleep(0.1)  # brief pause after reset
                continue

            # --- Arm PD control ---
            if is_valid and is_new:
                arm_target_q = sol_q.copy()

            if arm_target_q is not None:
                for i in range(7):
                    idx_l = LEFT_ARM_START + i
                    cur_pos_l = mj_data.qpos[arm_qpos_adr[i * 2]]
                    cur_vel_l = mj_data.qvel[arm_qvel_adr[i * 2]]
                    mj_data.ctrl[idx_l] = ARM_KP * (arm_target_q[i] - cur_pos_l) - ARM_KD * cur_vel_l

                    idx_r = RIGHT_ARM_START + i
                    cur_pos_r = mj_data.qpos[arm_qpos_adr[i * 2 + 1]]
                    cur_vel_r = mj_data.qvel[arm_qvel_adr[i * 2 + 1]]
                    mj_data.ctrl[idx_r] = ARM_KP * (arm_target_q[7 + i] - cur_pos_r) - ARM_KD * cur_vel_r

            # --- Fix floating base ---
            mj_data.qpos[:3] = initial_qpos[:3]
            mj_data.qpos[3:7] = initial_qpos[3:7]
            mj_data.qvel[:6] = 0.0

            # --- Leg + waist PD ---
            for i in range(15):
                mj_data.ctrl[i] = LEG_KP * (initial_qpos[leg_qpos_adr[i]] - mj_data.qpos[leg_qpos_adr[i]]) \
                                - LEG_KD * mj_data.qvel[leg_qvel_adr[i]]

            # --- Elastic band ---
            if config.ENABLE_ELASTIC_BAND:
                if elastic_band.enable:
                    mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                        mj_data.qpos[:3], mj_data.qvel[:3]
                    )

            mujoco.mj_step(mj_model, mj_data)

            locker.release()

            # --- Publish camera ---
            now = time.perf_counter()
            if cam_enabled and (now - last_cam_time) >= CAM_PUBLISH_INTERVAL:
                last_cam_time = now
                try:
                    renderer.update_scene(mj_data, camera=head_cam_id)
                    img = renderer.render()
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, jpg = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')
                    msg = json.dumps({"images": {"head_camera": b64}})
                    zmq_pub.send_string(msg, zmq.NOBLOCK)
                except Exception as e:
                    print(f"[Camera ERROR] {e}")

            time_until_next_step = mj_model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        except Exception as e:
            locker.release()
            print(f"[SimThread ERROR] {e}")
            import traceback
            traceback.print_exc()

    arm_reader.close()
    zmq_pub.close()
    zmq_ctx.term()


def PhysicsViewerThread():
    while viewer.is_running():
        try:
            locker.acquire()
            viewer.sync()
            locker.release()
            time.sleep(config.VIEWER_DT)
        except Exception as e:
            print(f"[ViewerThread ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
