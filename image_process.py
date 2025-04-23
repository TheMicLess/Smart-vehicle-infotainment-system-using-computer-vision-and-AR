from pyzbar import pyzbar
import cv2
import time
import multiprocessing
from multiprocessing import shared_memory, Lock
import numpy as np
from ultralytics import YOLOWorld

# Enable OpenCL in OpenCV
cv2.ocl.setUseOpenCL(True)

# Global variables
detected_qr = ""
last_input_time = time.time()
shm_ar_lock = multiprocessing.Lock()

def clear_ar_buffer(ar_buffer, data="CLR"):
    """Clear the AR buffer and optionally write data."""
    with shm_ar_lock:
        ar_buffer[:] = 0  # Clear previous data
        encoded_msg = data.encode("utf-8")
        ar_buffer[:len(encoded_msg)] = np.frombuffer(encoded_msg, dtype=np.uint8)

def process_qr_code(qr, gray_frame, ar_buffer, queue_sign_ids):
    """Process a single QR code."""
    global detected_qr, last_input_time

    (x, y, w, h) = qr.rect
    qr_data = qr.data.decode("utf-8")

    # Draw rectangle and label on the frame
    cv2.rectangle(gray_frame, (x, y), (x + w, y + h), 255, 2)
    cv2.putText(gray_frame, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

    if qr_data.startswith("AR"):
        encoded_msg = qr_data.encode("utf-8")[:256]
        points = np.array([qr.polygon[i] for i in range(4)], dtype=np.float32).flatten()
        points_bytes = points.tobytes()
        with shm_ar_lock:
            ar_buffer[:] = 0  # Clear previous data
            ar_buffer[:len(encoded_msg)] = np.frombuffer(encoded_msg, dtype=np.uint8)
            ar_buffer[256:256+32] = np.frombuffer(points_bytes, dtype=np.uint8)
    elif qr_data != detected_qr:
        clear_ar_buffer(ar_buffer)
        detected_qr = qr_data
        queue_sign_ids.put(qr_data)
        print(f"[INFO] Detected QR Code: {qr_data}")
        last_input_time = time.time()

def qr_decode(gray_frame, queue_sign_ids, ar_buffer):
    """Decode QR codes from the frame."""
    global last_input_time, detected_qr

    qrcodes = pyzbar.decode(gray_frame)
    if not qrcodes:
        clear_ar_buffer(ar_buffer)
    else:
        for qr in qrcodes:
            process_qr_code(qr, gray_frame, ar_buffer, queue_sign_ids)

    # Clear display if no input for 10 seconds
    if time.time() - last_input_time > 10:
        detected_qr = ""

    return gray_frame

def car_detect_haar_classifier(gray_frame, car_cascade, queue_info):
    """Detect cars using Haar Cascade."""
    cars = car_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=9, minSize=(60, 60))
    for (x, y, w, h) in cars:
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(gray_frame, "car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        queue_info.put("vehicle Detected")
        queue_info.put("slow down")
    return gray_frame

def car_detect_yolo(frame, model, queue_info):
    """Detect cars using YOLO model."""
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0].item()

            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if label.lower() in ["car", "truck", "bus", "person"]:
                    queue_info.put(f"{label.lower()} Detected")
                    queue_info.put("caution and slow")
    return frame

def initialize_shared_memory():
    """Initialize shared memory for video frames and AR info."""
    shm_vid = shared_memory.SharedMemory(name="video_frame")
    frame_array = np.ndarray((720, 1280, 3), dtype=np.uint8, buffer=shm_vid.buf)

    shm_AR = shared_memory.SharedMemory(name="ar_info")
    ar_buffer = np.ndarray((288,), dtype=np.uint8, buffer=shm_AR.buf)

    return shm_vid, frame_array, shm_AR, ar_buffer

def pi_cam_main(queue_sign_ids, queue_info, stop_event):
    """Main function for Pi camera processing."""
    url = "http://172.20.10.4:5000/video_feed"
    cap = cv2.VideoCapture(url)

    print("[INFO] QR Code detection started...")

    model = YOLOWorld(model='yolov8s-world.pt', verbose=False)
    model.set_classes(["car", "truck", "bus", "person"])

    shm_vid, frame_array, shm_AR, ar_buffer = initialize_shared_memory()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            #frame = car_detect_yolo(frame, model, queue_info)
            frame = cv2.resize(frame, (1280, 720))
            frame_array[:] = frame  # Copy frame to shared memory

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = qr_decode(gray, queue_sign_ids, ar_buffer)

            #cv2.imshow("QR & Car Detection", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()
        print("[INFO] Interrupted by user")
    finally:
        cap.release()
        shm_vid.close()
        shm_AR.close()
        cv2.destroyAllWindows()
