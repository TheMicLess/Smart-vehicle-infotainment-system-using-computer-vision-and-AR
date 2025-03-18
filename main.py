from qr_reader import detect_qr_codes
from hud_display import load_sign_images, display_signs

def main():
    sign_ids = detect_qr_codes()
    sign_ids = sign_ids.split('/')
    check = ""
    if sign_ids:
        if check != sign_ids: #avoiding multiple readings
            images = load_sign_images(sign_ids)
            display_signs(images)
            check = sign_ids

if __name__ == "__main__":
    main()
