import numpy as np
import cv2

def calibrate_camera():
    """Calibrate the camera and generate camera matrix and distortion coefficients."""
    chessboard_size = (8, 6)  # Number of inner corners per chessboard row and column
    square_size = 1.0  # Size of a square in your desired unit (e.g., meters, centimeters)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    images = ["1.jpg", "2.jpg"]  # Replace with your image paths

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Error: Unable to load image {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        raise ValueError("No valid chessboard corners were detected. Check your images and chessboard size.")

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        return camera_matrix, dist_coeffs
    else:
        raise ValueError("Camera calibration failed. Ensure you have valid calibration images.")

if __name__ == "__main__":
    try:
        camera_matrix, dist_coeffs = calibrate_camera()
        print("Camera Matrix (float32):")
        print(camera_matrix.astype(np.float32))
        print("\nDistortion Coefficients (float32):")
        print(dist_coeffs.astype(np.float32))
    except Exception as e:
        print(f"Error: {e}")
