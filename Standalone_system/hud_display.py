import cv2
import numpy as np
import os
import time
import threading
from queue import Empty 
import multiprocessing
from multiprocessing import shared_memory, Lock


lock = threading.Lock()
shm_ar_lock = multiprocessing.Lock()

# Global variables
sign_ids_temp_from_queue = ""
ar_temp_from_shm = ""
info_lines = []  # Stores last 5 lines for display

def load_sign_images(ids, sign_size=(50, 50)):
    """Load and resize sign images based on provided IDs."""
    ids = ids.split("//")
    images = []
    for sign_id in ids:
        filename = f"repo/{sign_id}.jpg"
        if os.path.exists(filename):
            img = cv2.imread(filename)
            img = cv2.resize(img, sign_size)
            images.append(img)
        else:
            print(f"Warning: {filename} not found")
    return images


def place_ar_image(frame, ar_image_path, location):
    """Place AR image on the frame at the specified location or remove it if no path is provided."""
    if not ar_image_path or not os.path.exists(f"{ar_image_path}.jpg"):
        # Clear the specified location on the frame
        return frame
    
    (x, y, w, h) = location
    frame[y:y+h, x:x+w] = 0  # Set the region to black

    img = cv2.imread(f"{ar_image_path}.jpg", cv2.IMREAD_UNCHANGED)
    (x, y, w, h) = location
    img_h, img_w = img.shape[:2]
    aspect_ratio = img_w / img_h

    # Resize image while maintaining aspect ratio
    new_w, new_h = (int(h * aspect_ratio), h) if w / h > aspect_ratio else (w, int(w / aspect_ratio))
    img_resized = cv2.resize(img, (new_w, new_h))

    # Compute position and crop to fit within frame bounds
    x1, x2 = max(0, x + (w - new_w) // 2), min(frame.shape[1], x + (w - new_w) // 2 + new_w)
    y1, y2 = max(0, y + (h - new_h) // 2), min(frame.shape[0], y + (h - new_h) // 2 + new_h)
    img_cropped = img_resized[: y2 - y1, : x2 - x1]
    roi = frame[y1:y2, x1:x2]

    # Handle transparency if present
    if img.shape[-1] == 4:
        overlay, mask = img_cropped[:, :, :3], img_cropped[:, :, 3]
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - mask / 255.0) + overlay[:, :, c] * (mask / 255.0)
    else:
        frame[y1:y2, x1:x2] = img_cropped

    return frame


def display_signs(images, info_lines, hud, window_size=( )):
    # Place images starting from top-left, stacking vertically
    x_offset, y_offset = 20, 20
    for img in images:
        if y_offset + img.shape[0] > window_size[1]:
            break  # Stop if window overflows
        hud[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
        y_offset += img.shape[0] + 10  

    # Display text info at top-right corner
    for i, line in enumerate(info_lines):
        text_x, text_y = window_size[0] - (10 * len(line)), 20  
        
        cv2.putText(hud, line, (text_x, text_y + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
         
    cv2.imshow("HUD Display", hud)
    cv2.waitKey(1)


def queue_reader(queue_sign_ids, queue_info, stop_event):
    global sign_ids_temp_from_queue, info_lines
    last_info_time = time.time()
    while not stop_event.is_set():
        
        try:
            with lock:
                temp = queue_sign_ids.get(block=False)
                sign_ids_temp_from_queue = temp
        except Empty:
            pass
        
        try:
            with lock:
                new_info = queue_info.get(block=False)
                if new_info not in info_lines:
                    info_lines.append(new_info)
                    
                if len(info_lines) > 5:
                    info_lines.pop(0)
        except Empty:
            pass
        
        with lock:
            if info_lines and (time.time() - last_info_time > 10):
                info_lines.pop(0)
                last_info_time = time.time()  # Reset timer after popping
        time.sleep(0.1)
    

def hud_main(queue_sign_ids, queue_info, stop_event, frame_queue):
    global sign_ids_temp_from_queue, info_lines, ar_temp_from_shm
    last_input_time = time.time()
    displayed_images = []
    x = y = w = h = 0

    while True:
        try:
            shm_AR = shared_memory.SharedMemory(name="ar_info")
            break  
        except FileNotFoundError:
            print("Waiting for shared memory...")
            time.sleep(0.5)

    while True:
        try:
            shm_vid = shared_memory.SharedMemory(name="video_frame")
            break  
        except FileNotFoundError:
            print("Waiting for shared memory...")
            time.sleep(0.5)  

    frame_array_buffer = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm_vid.buf)
    ar_buffer = np.ndarray((51,), dtype=np.uint8, buffer=shm_AR.buf)    

    # Start input listener in a separate thread
    input_thread = threading.Thread(target=queue_reader, args=(queue_sign_ids, queue_info, stop_event), daemon=True)
    input_thread.start()
    
    try:
        while not stop_event.is_set():
            ar_image_path = ""

            with shm_ar_lock:
                ar_temp_from_shm = ar_buffer.tobytes().decode("utf-8").rstrip("\x00")
            
            if ar_temp_from_shm[0:2] == "AR":
                _, ar_image_path , location  = ar_temp_from_shm.split("//")
                x, y, w, h = map(int, location.split("/"))
                
            else:  
                ar_image_path = ""
                x = y = w = h = 0
                            
            if sign_ids_temp_from_queue:
                sign_ids = sign_ids_temp_from_queue
                displayed_images = load_sign_images(sign_ids)
                last_input_time = time.time()
                sign_ids_temp_from_queue = ""  # Reset after processing

            # Clear display if (10 seconds) have passed without input
            if time.time() - last_input_time > 10:
                displayed_images = []
                
            frame = frame_array_buffer.copy() 
            frame = place_ar_image(frame, ar_image_path, (x, y, w, h)) 
            display_signs(displayed_images, info_lines, frame)
            time.sleep(0.1)  # Small delay to reduce CPU usage

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        shm_vid.close()
        shm_AR.close()
        cv2.destroyAllWindows()
        print("Exiting program...")
        