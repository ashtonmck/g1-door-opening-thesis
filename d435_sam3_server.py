import asyncio
import websockets
import cv2
import numpy as np
import torch
import sys
import time
import json
import threading
from PIL import Image

sys.path.insert(0, '/home/rppl/sam3')
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import pyrealsense2 as rs

print("Loading SAM3...")
model = build_sam3_image_model(
    bpe_path='/home/rppl/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
    device='cuda',
    eval_mode=True,
    checkpoint_path='/home/rppl/sam3_weights/sam3.pt',
    load_from_HF=False,
)
processor = Sam3Processor(model, device='cuda')
print("SAM3 ready")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("D435 started")

# All objects to detect
PROMPTS = ["door", "hand", "guitar"]
COLORS = {
    "door": (0, 255, 0),    # green
    "hand": (255, 0, 0),          # blue
    "guitar": (0, 0, 255),           # red
}

latest_frame = None
detected_objects = {}  # prompt -> mask
selected_object = None
frame_count = 0
current_prompt_idx = 0

def capture_and_process():
    global latest_frame, detected_objects, frame_count, current_prompt_idx

    while True:
        try:
            frames = pipeline.wait_for_frames(1000)
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            frame_count += 1

            # Cycle through prompts - one per inference cycle
            if frame_count % 10 == 0:
                prompt = PROMPTS[current_prompt_idx % len(PROMPTS)]
                current_prompt_idx += 1

                
                img_pil = Image.fromarray(img_rgb)

                t0 = time.time()
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                    state = processor.set_image(img_pil)
                    state = processor.set_text_prompt(prompt, state)
                dt = time.time() - t0

                masks = state.get("masks")
                if masks is not None and len(masks) > 0:
                    m = masks[0]
                    if isinstance(m, torch.Tensor):
                        m = m.cpu().numpy()
                    m = m.squeeze()
                    
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.float32), (w, h),
                                       interpolation=cv2.INTER_NEAREST)
                                       
                    mask = (m > 0.5).astype(bool)
                    
                    if mask.sum() > 10:
                        mask_u8 = mask.astype(np.uint8) * 255
                        
                        # --- NEW: Blur the mask to create a perfectly smooth, rounded edge ---
                        mask_u8 = cv2.GaussianBlur(mask_u8, (7, 7), 0)
                        _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
                        
                        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            
                            clean_mask = np.zeros_like(mask_u8)
                            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                            
                            detected_objects[prompt] = clean_mask > 0
                            
                            # Save the smooth contour shape instead of a blocky bounding box
                            detected_objects[prompt + "_contour"] = largest_contour
                            
                            print(f"[{dt:.2f}s] FOUND '{prompt}': {mask.sum()} px")
                        else:
                            detected_objects.pop(prompt, None)
                            detected_objects.pop(prompt + "_contour", None)
                    else:
                        detected_objects.pop(prompt, None)
                        detected_objects.pop(prompt + "_contour", None)
                        print(f"[{dt:.2f}s] '{prompt}': too few pixels")
                else:
                    detected_objects.pop(prompt, None)
                    detected_objects.pop(prompt + "_contour", None)
                    print(f"[{dt:.2f}s] NOT FOUND: '{prompt}'")

            # Compose display frame with all detected objects
            display = img.copy()
            obj_list = []
            for prompt in PROMPTS:
                if prompt in detected_objects:
                    mask = detected_objects[prompt]
                    contour = detected_objects.get(prompt + "_contour")
                    color_rgb = COLORS.get(prompt, (0, 255, 0))
                    
                    # 1. Translucent fill perfectly mapped to the object's surface
                    opacity = 0.6 if prompt == selected_object else 0.35
                    display[mask] = (display[mask] * (1 - opacity) +
                                     np.array(color_rgb) * opacity).astype(np.uint8)
                    
                    if contour is not None:
                        # 2. Draw a smooth, anti-aliased outline hugging the shape (cv2.LINE_AA)
                        thickness = 4 if prompt == selected_object else 2
                        cv2.drawContours(display, [contour], -1, color_rgb, thickness, cv2.LINE_AA)
                        
                        # 3. Calculate the top edge of the shape to anchor the label
                        x, y, w, h = cv2.boundingRect(contour)
                        label = f"{prompt}" + (" [SELECTED]" if prompt == selected_object else "")
                        
                        # Use LINE_AA on the text as well so it looks clean
                        cv2.putText(display, label, (x, max(20, y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    color_rgb, 2, cv2.LINE_AA)

            # Status bar
            status = f"Detected: {len([p for p in PROMPTS if p in detected_objects])}/{len(PROMPTS)}"
            if selected_object:
                status += f" | Selected: {selected_object}"
            cv2.putText(display, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            _, jpg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            latest_frame = {
                "image": jpg.tobytes(),
                "objects": obj_list
            }
            cv2.imshow("SAM3 Live Feed", display)
            
            # Wait for key press for 1 millisecond. If 's' is pressed, save it!
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                filename = f"sam3_desktop_shot_{int(time.time())}.jpg"
                cv2.imwrite(filename, display)
                print(f"[Server] 📸 SCREENSHOT SAVED TO DISK: {filename}")

        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(0.1)


async def handle(websocket):
    global selected_object
    print(f"[Quest] Client connected: {websocket.remote_address}")
    
    async def receive_loop():
        global selected_object, PROMPTS
        try:
            async for msg in websocket:
                try:
                    data = json.loads(msg)
                    if "select" in data:
                        selected_object = data["select"]
                        print(f"[Quest] SELECTED: '{selected_object}'")
                        print(f"[Quest] Would execute policy for: {selected_object}")
                        if latest_frame is not None:
                            filename = f"photo_{selected_object}_{int(time.time())}.jpg"
                            with open(filename, "wb") as f:
                                f.write(latest_frame["image"])
                            print(f"[Server] 📸 SCREENSHOT SAVED: {filename}")
                    if "prompts" in data:
                        PROMPTS = data["prompts"]
                        print(f"[Quest] Updated prompts: {PROMPTS}")
                except Exception as e:
                    print(f"[Quest] Receive parse error: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def send_loop():
        try:
            while True:
                if latest_frame is not None:
                    await websocket.send(latest_frame["image"])
                await asyncio.sleep(0.05)  # 20fps stream
        except websockets.exceptions.ConnectionClosed:
            pass
    
    try:
        await asyncio.gather(receive_loop(), send_loop())
    except websockets.exceptions.ConnectionClosed:
        print("[Quest] Client disconnected")


async def main():
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    t = threading.Thread(target=capture_and_process, daemon=True)
    t.start()

    print(f"[Server] ws://{local_ip}:8765")
    print(f"[Server] Detecting: {PROMPTS}")
    print("[Server] Ready for Quest connection")

    async with websockets.serve(handle, "0.0.0.0", 8765, max_size=10**7):
        await asyncio.Future()

asyncio.run(main())
