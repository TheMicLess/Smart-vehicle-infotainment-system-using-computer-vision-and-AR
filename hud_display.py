import cv2
import numpy as np
import os

def load_sign_images(ids, sign_size=(50, 50)):
    images = []
    for sign_id in ids:
        filename = f"repo/1/{sign_id}.jpg"
        if os.path.exists(filename):
            img = cv2.imread(filename)
            img = cv2.resize(img, sign_size)
            images.append(img)
        else:
            print(f"Warning: {filename} not found")
    return images

def display_signs(images, window_size=(600, 200)):
    if not images:
        print("No images to display.")
        return
    
    hud = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
    x_offset = 20
    y_offset = (window_size[1] - images[0].shape[0]) // 2
    
    for img in images:
        if x_offset + img.shape[1] > window_size[0]:
            break
        
        hud[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        x_offset += img.shape[1] + 20
    
    cv2.imshow("HUD Display", hud)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
