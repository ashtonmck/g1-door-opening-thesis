# Launch Commands

## Door Opening Task

### Terminal 1 - Sim
conda activate unitree_sim_env
export CYCLONEDDS_URI=file:///home/rppl/cyclonedds.xml
cd ~/unitree_sim_isaaclab
python sim_main.py --device cuda --robot_type g129 \
  --task Isaac-DoorOpening-G129-Dex3-Joint \
  --enable_cameras --enable_dex3_dds

### Terminal 2 - Teleop (with recording)
conda activate tv
export CYCLONEDDS_URI=file:///home/rppl/cyclonedds.xml
cd ~/xr_teleoperate/teleop
python teleop_hand_and_arm.py \
  --input-mode=hand --arm=G1_29 --ee=dex3 --sim \
  --network-interface lo --img-server-ip YOUR_IP \
  --display-mode pass-through --record \
  --task-name "door_opening"

### Quest browser
https://YOUR_IP:8012?ws=wss://YOUR_IP:8012

## SSL cert (when IP changes)
mkcert -cert-file cert.pem -key-file key.pem YOUR_IP localhost 127.0.0.1
cp cert.pem key.pem ~/.config/xr_teleoperate/
cd $(mkcert -CAROOT) && python3 -m http.server 8888

## Controls
r → sync hands
s → start/stop recording
q → quit
