"""
MuJoCo Camera Bridge
Converts MuJoCo ZMQ camera stream (base64 JSON on port 5555)
to raw JPEG bytes (on port 55555) for xr_teleoperate.

Run this AFTER run_sim.py and BEFORE teleop_hand_and_arm.py.

Usage:
    python mujoco_camera_bridge.py
"""

import zmq
import json
import base64
import time

# MuJoCo sim publishes on this port (base64 JSON format)
MUJOCO_ZMQ_PORT = 5555

# xr_teleoperate subscribes on these ports (raw JPEG format)
HEAD_CAMERA_PORT   = 55555
LEFT_WRIST_PORT    = 55556
RIGHT_WRIST_PORT   = 55557

# Camera key in MuJoCo's JSON message
HEAD_CAMERA_KEY = "head_camera"

def main():
    ctx = zmq.Context()

    # Subscribe to MuJoCo sim stream
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{MUJOCO_ZMQ_PORT}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 1000)

    # Publish raw JPEG for head camera
    head_pub = ctx.socket(zmq.PUB)
    head_pub.bind(f"tcp://*:{HEAD_CAMERA_PORT}")

    # Publish blank frames for wrist cameras (not available in MuJoCo yet)
    left_pub = ctx.socket(zmq.PUB)
    left_pub.bind(f"tcp://*:{LEFT_WRIST_PORT}")

    right_pub = ctx.socket(zmq.PUB)
    right_pub.bind(f"tcp://*:{RIGHT_WRIST_PORT}")

    print(f"[Bridge] Subscribed to MuJoCo on port {MUJOCO_ZMQ_PORT}")
    print(f"[Bridge] Publishing head camera on port {HEAD_CAMERA_PORT}")
    print(f"[Bridge] Publishing (blank) wrist cameras on ports {LEFT_WRIST_PORT}, {RIGHT_WRIST_PORT}")
    print("[Bridge] Running... Ctrl+C to stop")

    # Generate a blank JPEG frame for missing wrist cameras
    import cv2
    import numpy as np
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "No wrist camera", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
    _, blank_jpg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 80])
    blank_bytes = blank_jpg.tobytes()

    fps_count = 0
    fps_time = time.time()

    while True:
        try:
            raw = sub.recv()
            msg = json.loads(raw.decode("utf-8"))

            images = msg.get("images", {})

            # Head camera
            if HEAD_CAMERA_KEY in images:
                jpg_bytes = base64.b64decode(images[HEAD_CAMERA_KEY])
                head_pub.send(jpg_bytes)
            
            # Wrist cameras - send blank frames so xr_teleoperate doesn't hang
            left_pub.send(blank_bytes)
            right_pub.send(blank_bytes)

            fps_count += 1
            now = time.time()
            if now - fps_time >= 5.0:
                print(f"[Bridge] FPS: {fps_count / (now - fps_time):.1f}")
                fps_count = 0
                fps_time = now

        except zmq.Again:
            print("[Bridge] Waiting for MuJoCo sim...")
        except KeyboardInterrupt:
            print("[Bridge] Stopping.")
            break
        except Exception as e:
            print(f"[Bridge] Error: {e}")
            time.sleep(0.1)

    sub.close()
    head_pub.close()
    left_pub.close()
    right_pub.close()
    ctx.term()

if __name__ == "__main__":
    main()
