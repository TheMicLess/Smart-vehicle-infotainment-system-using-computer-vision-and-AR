from picamera2 import Picamera2
from pyzbar import pyzbar
import cv2
import time

def detect_qr_codes():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480)"format": "RGB888"},
        
    )
    """config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 60}
    )"""
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    
    print("[INFO] QR Code detection started...")
    try:
        while True:
            frame = picam2.capture_array("main")
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            qrcodes = pyzbar.decode(gray)
            
            sign_ids = ""
            for qr in qrcodes:
                qr_data = qr.data.decode("utf-8")
                #sign_ids.append(qr_data)
                sign_ids = qr_data
                cv2.rectangle(gray, (qr.rect.left, qr.rect.top), 
                              (qr.rect.left + qr.rect.width, qr.rect.top + qr.rect.height), 255, 2)
                cv2.putText(gray, qr_data, (qr.rect.left, qr.rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                print(f"[INFO] Detected QR Code: {qr_data}")
            
            cv2.imshow("QR Code Scanner", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            if sign_ids:
                return sign_ids
    
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

