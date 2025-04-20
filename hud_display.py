import cv2
import numpy as np
import os
import time
import threading
from queue import Empty
import multiprocessing
from multiprocessing import shared_memory, Lock
from testing.ar_model_generater import *
from ar_model_generater_with_shaders import *

cv2.ocl.setUseOpenCL(True)

# Locks for thread and shared memory synchronization
lock = threading.Lock()
shm_ar_lock = multiprocessing.Lock()

# Global variables
sign_ids_temp_from_queue = ""
ar_temp_from_shm = ""
info_lines = []

# Constants
SIGN_SIZE = (100, 100)
WINDOW_SIZE = (1280, 720)
INFO_DISPLAY_DURATION = 10  # seconds
MAX_INFO_LINES = 5


def load_sign_images(ids, sign_size=SIGN_SIZE):
    """Load sign images based on IDs."""
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


def display_signs(images, info_lines, hud, window_size=WINDOW_SIZE):
    """Display images and text information on the HUD."""
    x_offset, y_offset = 20, 20

    # Display images
    for img in images:
        if y_offset + img.shape[0] > window_size[1]:
            break  # Stop if window overflows
        hud[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
        y_offset += img.shape[0] + 10

    # Display text info
    for i, line in enumerate(info_lines):
        text_x = window_size[0] - (10 * len(line))
        text_y = 20 + i * 30
        cv2.putText(hud, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("HUD Display", hud)
    cv2.waitKey(1)


def queue_reader(queue_sign_ids, queue_info, stop_event):
    """Read data from queues and update global variables."""
    global sign_ids_temp_from_queue, info_lines
    last_info_time = time.time()

    while not stop_event.is_set():
        # Read sign IDs
        try:
            with lock:
                sign_ids_temp_from_queue = queue_sign_ids.get(block=False)
        except Empty:
            pass

        # Read info lines
        try:
            with lock:
                new_info = queue_info.get(block=False)
                if new_info not in info_lines:
                    info_lines.append(new_info)
                if len(info_lines) > MAX_INFO_LINES:
                    info_lines.pop(0)
        except Empty:
            pass

        # Remove old info lines
        with lock:
            if info_lines and (time.time() - last_info_time > INFO_DISPLAY_DURATION):
                info_lines.pop(0)
                last_info_time = time.time()

        time.sleep(0.1)


def initialize_shared_memory(name, shape, dtype):
    """Initialize shared memory and return a numpy array."""
    while True:
        try:
            shm = shared_memory.SharedMemory(name=name)
            return np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm
        except FileNotFoundError:
            print(f"Waiting for shared memory: {name}...")
            time.sleep(0.5)


def hud_main(queue_sign_ids, queue_info, stop_event):
    """Main function to handle the HUD display."""
    global sign_ids_temp_from_queue, info_lines
    last_input_time = time.time()
    displayed_images = []

    # Initialize shared memory
    frame_array_buffer, shm_vid = initialize_shared_memory("video_frame", (720, 1280, 3), np.uint8)
    ar_buffer, shm_AR = initialize_shared_memory("ar_info", (288,), np.uint8)

    # Start input listener in a separate thread
    input_thread = threading.Thread(target=queue_reader, args=(queue_sign_ids, queue_info, stop_event), daemon=True)
    input_thread.start()

    try:
        while not stop_event.is_set():
            frame = frame_array_buffer.copy()

            # Process AR data
            with shm_ar_lock:
                raw_data = bytes(ar_buffer[:256])
                qr_data_str = raw_data.split(b'\x00', 1)[0].decode('utf-8')
                points_bytes = bytes(ar_buffer[256:256 + 32])
                points = np.frombuffer(points_bytes, dtype=np.float32).reshape((4, 2))

            if qr_data_str.startswith("AR"):
                #_, qr_data_str, size = qr_data_str.split("//")
                _, qr_data_str= qr_data_str.split("//")

                #frame = generate_3d_model_mlt(frame, qr_data_str, points, float(size))
                frame = generate_3d_model_mlt(frame, qr_data_str, points)

            # Load and display sign images
            if sign_ids_temp_from_queue:
                displayed_images = load_sign_images(sign_ids_temp_from_queue)
                last_input_time = time.time()
                sign_ids_temp_from_queue = ""  # Reset after processing

            # Clear display if no input for a while
            if time.time() - last_input_time > INFO_DISPLAY_DURATION:
                displayed_images = []

            display_signs(displayed_images, info_lines, frame)
            time.sleep(0.1)  # Small delay to reduce CPU usage

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()  # Signal to stop if 'q' is pressed
                break

    except KeyboardInterrupt:
        stop_event.set()
        print("[INFO] Interrupted by user")

    finally:
        shm_vid.close()
        shm_AR.close()
        cv2.destroyAllWindows()
        print("Exiting program...")
