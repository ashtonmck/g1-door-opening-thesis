import numpy as np
import mujoco
from mujoco import viewer as mj_viewer
import time, os

scene = os.path.expanduser('~/unitree_mujoco/unitree_robots/g1/scene_29dof_door.xml')
mj_model = mujoco.MjModel.from_xml_path(scene)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.001
initial_qpos = mj_data.qpos.copy()

LEFT_ARM_START, RIGHT_ARM_START = 15, 22
arm_qpos_adr, arm_qvel_adr = [], []
for i in range(7):
    jl = mj_model.actuator_trnid[LEFT_ARM_START+i, 0]
    arm_qpos_adr.append(mj_model.jnt_qposadr[jl])
    arm_qvel_adr.append(mj_model.jnt_dofadr[jl])
    jr = mj_model.actuator_trnid[RIGHT_ARM_START+i, 0]
    arm_qpos_adr.append(mj_model.jnt_qposadr[jr])
    arm_qvel_adr.append(mj_model.jnt_dofadr[jr])

leg_qpos_adr, leg_qvel_adr = [], []
for i in range(15):
    j = mj_model.actuator_trnid[i, 0]
    leg_qpos_adr.append(mj_model.jnt_qposadr[j])
    leg_qvel_adr.append(mj_model.jnt_dofadr[j])

v = mj_viewer.launch_passive(mj_model, mj_data)

traj = np.load(os.path.expanduser('~/eval_results/traj_ep1.npy'))

SHOW_STEPS = 12
INTERP_STEPS = 20
ARM_KP, ARM_KD = 80.0, 6.0
LEG_KP, LEG_KD = 200.0, 10.0

mj_data.qpos[:] = initial_qpos
mj_data.qvel[:] = 0.0
mj_data.ctrl[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
v.sync()

print("="*50)
print("  Set up your camera angle in the viewer.")
print("  Start your screen recorder.")
print("  Press Enter here when ready...")
print("="*50)
input()

print("Starting in 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("GO!")

prev_lt = np.array([initial_qpos[arm_qpos_adr[i*2]] for i in range(7)])
prev_rt = np.array([initial_qpos[arm_qpos_adr[i*2+1]] for i in range(7)])

for step in range(SHOW_STEPS):
    saved = traj[step]
    next_lt = np.array([saved[arm_qpos_adr[i*2]] for i in range(7)])
    next_rt = np.array([saved[arm_qpos_adr[i*2+1]] for i in range(7)])

    for sub in range(INTERP_STEPS):
        alpha = sub / INTERP_STEPS
        lt = prev_lt + alpha * (next_lt - prev_lt)
        rt = prev_rt + alpha * (next_rt - prev_rt)

        for i in range(7):
            mj_data.ctrl[LEFT_ARM_START+i] = ARM_KP*(lt[i]-mj_data.qpos[arm_qpos_adr[i*2]]) - ARM_KD*mj_data.qvel[arm_qvel_adr[i*2]]
            mj_data.ctrl[RIGHT_ARM_START+i] = ARM_KP*(rt[i]-mj_data.qpos[arm_qpos_adr[i*2+1]]) - ARM_KD*mj_data.qvel[arm_qvel_adr[i*2+1]]

        mj_data.qpos[:3] = initial_qpos[:3]
        mj_data.qpos[3:7] = initial_qpos[3:7]
        mj_data.qvel[:6] = 0.0
        for i in range(15):
            mj_data.ctrl[i] = LEG_KP*(initial_qpos[leg_qpos_adr[i]]-mj_data.qpos[leg_qpos_adr[i]]) - LEG_KD*mj_data.qvel[leg_qvel_adr[i]]

        for _ in range(5):
            mujoco.mj_step(mj_model, mj_data)

        v.sync()
        time.sleep(0.01)

    prev_lt = next_lt.copy()
    prev_rt = next_rt.copy()

    dj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'door_hinge')
    angle = np.degrees(mj_data.qpos[mj_model.jnt_qposadr[dj]])
    print(f"  step {step}: door={angle:.1f}")

final_lt = prev_lt.copy()
final_rt = prev_rt.copy()
print("\nHolding for 5 seconds...")
for _ in range(500):
    for i in range(7):
        mj_data.ctrl[LEFT_ARM_START+i] = ARM_KP*(final_lt[i]-mj_data.qpos[arm_qpos_adr[i*2]]) - ARM_KD*mj_data.qvel[arm_qvel_adr[i*2]]
        mj_data.ctrl[RIGHT_ARM_START+i] = ARM_KP*(final_rt[i]-mj_data.qpos[arm_qpos_adr[i*2+1]]) - ARM_KD*mj_data.qvel[arm_qvel_adr[i*2+1]]

    mj_data.qpos[:3] = initial_qpos[:3]
    mj_data.qpos[3:7] = initial_qpos[3:7]
    mj_data.qvel[:6] = 0.0
    for i in range(15):
        mj_data.ctrl[i] = LEG_KP*(initial_qpos[leg_qpos_adr[i]]-mj_data.qpos[leg_qpos_adr[i]]) - LEG_KD*mj_data.qvel[leg_qvel_adr[i]]

    for _ in range(5):
        mujoco.mj_step(mj_model, mj_data)
    v.sync()
    time.sleep(0.01)

print("Done. Close viewer window.")
while v.is_running():
    time.sleep(0.1)
