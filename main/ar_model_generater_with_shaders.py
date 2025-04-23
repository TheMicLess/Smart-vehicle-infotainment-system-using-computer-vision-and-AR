import cv2
import numpy as np
import torch
import os

# Constants
DEFAULT_COLOR = (200, 200, 200)
ROTATION_ANGLE_DEGREES = 90
CAMERA_MATRIX = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))

# Global variables
vertices = []
faces = []
loaded_qr_data_str = ""
resize = 0
mtl_filename = ""

cv2.ocl.setUseOpenCL(True)


def load_mtl(filename):
    """Load material properties from an MTL file."""
    materials = {}
    current_material = None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('newmtl'):
                current_material = line.split()[1]
                materials[current_material] = {'Kd': (1.0, 1.0, 1.0)}
            elif line.startswith('Kd') and current_material:
                kd = list(map(float, line.split()[1:4]))
                kd_bgr = tuple(int(c * 255) for c in kd[::-1])
                materials[current_material]['Kd'] = kd_bgr
    return materials


def load_obj_with_mtl(obj_path):
    """Load OBJ file along with its MTL file."""
    global resize
    vertices = []
    faces = []
    face_materials = []
    mtl_path = None

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('mtllib'):
                mtl_path = line.split()[1].strip()
            elif line.startswith('usemtl'):
                current_mtl = line.split()[1].strip() if len(line.split()) > 1 else None
            elif line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):
                face = [int(p.split('/')[0]) - 1 for p in line.strip().split()[1:]]
                faces.append(face)
                face_materials.append(current_mtl)

    vertices_np = np.array(vertices)
    vertices[:] = (vertices_np * resize).tolist()
    return vertices, faces, face_materials, mtl_path


def check_file_exists(filepath, file_type):
    """Check if a file exists and print an error message if it doesn't."""
    if not os.path.exists(filepath):
        print(f"{file_type} file not found: {filepath}")
        return False
    return True


def apply_rotation(points, angle_degrees):
    """Apply rotation to 3D points."""
    angle_rad = np.deg2rad(angle_degrees)
    rotation_matrix_y = np.array([
        [np.sin(angle_rad), 0, np.cos(angle_rad)],
        [0, 1, 0],
        [np.cos(angle_rad), 0, np.sin(angle_rad)]
    ], dtype=np.float32)
    return points @ rotation_matrix_y.T


def render_faces(frame, faces, camera_space_vertices, face_materials, materials, rvecs, tvecs):
    """Render faces on the frame."""
    face_depths = [
        (np.mean(camera_space_vertices[face][:, 2]), i)
        for i, face in enumerate(faces)
    ]
    face_depths.sort(reverse=True, key=lambda x: x[0])

    for _, face_index in face_depths:
        face = faces[face_index]
        mat_name = face_materials[face_index]
        pts_3d = np.array([vertices[i] for i in face], dtype=np.float32)
        pts_3d = apply_rotation(pts_3d, ROTATION_ANGLE_DEGREES)

        pts_2d, _ = cv2.projectPoints(pts_3d, rvecs, tvecs, CAMERA_MATRIX, DIST_COEFFS)
        pts_2d = np.int32(pts_2d).reshape(-1, 2)

        color = materials.get(mat_name, {}).get('Kd', DEFAULT_COLOR)
        cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 0, 0), thickness=2)
        cv2.fillPoly(frame, [pts_2d], color=color)


def generate_3d_model_mlt(frame, qr_data_str, points, size=0.025):
    """Generate a 3D model and render it on the frame."""
    global vertices, faces, face_materials, loaded_qr_data_str, resize, mtl_filename
        
    if qr_data_str != loaded_qr_data_str:
        resize = size
        obj_file = f"models/{qr_data_str}.obj"

        if not check_file_exists(obj_file, "OBJ"):
            return frame

        vertices, faces, face_materials, mtl_filename = load_obj_with_mtl(obj_file)
        mtl_file = f"models/{mtl_filename}"

        if not check_file_exists(mtl_file, "Material"):
            return frame

        materials = load_mtl(mtl_file)
        loaded_qr_data_str = qr_data_str
    else:
        materials = load_mtl(f"models/{mtl_filename}")

    obj_points = np.array([
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0]
    ], dtype=np.float32)

    success, rvecs, tvecs = cv2.solvePnP(obj_points, points, CAMERA_MATRIX, DIST_COEFFS)
    if not success:
        return frame

    rotation_matrix, _ = cv2.Rodrigues(rvecs)
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    rotation_matrix_tensor = torch.tensor(rotation_matrix, dtype=torch.float32).to(vertices_tensor.device)

    camera_space_vertices = torch.matmul(vertices_tensor, rotation_matrix_tensor.T).cpu().numpy() + tvecs.T
    render_faces(frame, faces, camera_space_vertices, face_materials, materials, rvecs, tvecs)

    return frame
