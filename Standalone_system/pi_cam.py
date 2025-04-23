from picamera2 import Picamera2
from pyzbar import pyzbar
import cv2
import time
import multiprocessing
from multiprocessing import shared_memory, Lock
import numpy as np

detected_qr = ""
last_input_time = time.time()
shm_ar_lock = multiprocessing.Lock()

def qr_decode(gray_frame, queue_sign_ids, ar_buffer):
    global last_input_time, detected_qr
    qrcodes = pyzbar.decode(gray_frame)
    if len(qrcodes) == 0:
        ar_data = "CLR"
        with shm_ar_lock:  
            ar_buffer[:] = 0  # Clear previous data
            encoded_msg = ar_data.encode("utf-8")
            ar_buffer[:len(encoded_msg)] = np.frombuffer(encoded_msg, dtype=np.uint8)
    else:
        for qr in qrcodes:
            (x, y, w, h) = qr.rect
            cv2.rectangle(gray_frame, (x, y), (x + w, y + h), 255, 2)
            qr_data = qr.data.decode("utf-8")
            cv2.putText(gray_frame, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            if qr_data[0:2] == "AR":
                ar_data = qr_data+f"//{x}/{y}/{w}/{h}"
                with shm_ar_lock:  
                    ar_buffer[:] = 0  # Clear previous data
                    encoded_msg = ar_data.encode("utf-8")
                    ar_buffer[:len(encoded_msg)] = np.frombuffer(encoded_msg, dtype=np.uint8)
                #print(f"[INFO] Detected QR Code: {ar_data}")
                
            elif qr_data != detected_qr:
                ar_data = "CLR"
                with shm_ar_lock:  
                    ar_buffer[:] = 0  # Clear previous data
                    encoded_msg = ar_data.encode("utf-8")
                    ar_buffer[:len(encoded_msg)] = np.frombuffer(encoded_msg, dtype=np.uint8)

                detected_qr = qr_data  # Avoiding multiple readings of the same QR code
                queue_sign_ids.put(qr_data)
                print(f"[INFO] Detected QR Code: {qr_data}")
                last_input_time = time.time()  # Update only when a QR code is detected
    

    # Clear display if (10 seconds) have passed without input
    if time.time() - last_input_time > 10:
        detected_qr = ""
        
    return gray_frame


def car_detect(gray_frame, car_cascade, queue_info):
    cars = car_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=9, minSize=(40, 40))
    for (x,y,w,h) in cars:
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(gray_frame, "car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
        queue_info.put(f"vehicle Detected")
        queue_info.put("slow down")
    return gray_frame


def pi_cam_main(queue_sign_ids, queue_info, stop_event, frame_queue):  
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},  
    controls={"FrameRate": 60}  # Maximize frame rate
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    
    print("[INFO] QR Code detection started...")
    
    # Load the pre-trained Haar Cascade for car detection
    car_cascade = cv2.CascadeClassifier("cars.xml")
    
    shm_vid = shared_memory.SharedMemory(name="video_frame")  # Access shared memory
    frame_array = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm_vid.buf)
    
    shm_AR = shared_memory.SharedMemory(name="ar_info")
    ar_buffer = np.ndarray((51,), dtype=np.uint8, buffer=shm_AR.buf)

    try:
        while not stop_event.is_set():  # Use stop_event to control the loop
            frame = picam2.capture_array("main")
            frame_array[:] = frame  # Copy frame directly to shared memory
            
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            gray = qr_decode(gray, queue_sign_ids, ar_buffer) 
            
            gray = car_detect(gray, car_cascade, queue_info)

            cv2.imshow("QR & Car Detection", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()  # Signal to stop if 'q' is pressed
                break
    
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        shm_vid.close()  
        shm_AR.close()
        cv2.destroyAllWindows()
        picam2.stop()
