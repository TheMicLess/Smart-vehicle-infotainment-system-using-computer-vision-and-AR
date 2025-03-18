import threading
import cv2
import time

# Function to handle QR detection
def qr_detection():
    cap = cv2.VideoCapture(0)
    qr_detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, decoded_info, _, _ = qr_detector.detectAndDecodeMulti(frame)
        if decoded_info:
            print(f"QR Code detected: {decoded_info}")
        cv2.imshow('QR Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle HUD display
def hud_display():
    while True:
        print("HUD Display running...")
        time.sleep(1)
        # Add your HUD display code here

# Create threads for QR detection and HUD display
qr_thread = threading.Thread(target=qr_detection)
hud_thread = threading.Thread(target=hud_display)

# Start the threads
qr_thread.start()
hud_thread.start()

# Wait for the threads to complete
qr_thread.join()
hud_thread.join()
