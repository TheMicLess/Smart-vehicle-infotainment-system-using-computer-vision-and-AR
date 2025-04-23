from flask import Flask, Response
import cv2
from picamera2 import Picamera2
import time

app = Flask(__name__)


picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888"},  
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()

def generate_frames():
    while True:
        try:
            frame = picam2.capture_array("main")
            if frame is None:
                continue

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames(): {e}")
            break

@app.route('/')
def index():
    return "<h1>Live Camera Feed</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
    finally:
        picam2.stop()
