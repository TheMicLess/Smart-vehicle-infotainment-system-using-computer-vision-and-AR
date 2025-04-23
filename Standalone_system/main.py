import multiprocessing
from multiprocessing import shared_memory
from pi_cam import pi_cam_main
from hud_display import hud_main

def main():
    queue_sign_ids = multiprocessing.Queue()
    queue_info = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    shm_vid = shared_memory.SharedMemory(name="video_frame", create=True, size=640 * 480 * 3)
    shm_AR = shared_memory.SharedMemory(name="ar_info", create=True, size=51)

    process_QR = multiprocessing.Process(target=pi_cam_main, args=(queue_sign_ids, queue_info, stop_event, None))
    process_HUD = multiprocessing.Process(target=hud_main, args=(queue_sign_ids, queue_info, stop_event, None))

    try:
        process_QR.start()
        process_HUD.start()
        process_QR.join()
        process_HUD.join()
    except KeyboardInterrupt:
        stop_event.set()
        process_QR.terminate()
        process_HUD.terminate()
    finally:
        shm_vid.close()
        shm_vid.unlink()
        shm_AR.close()
        shm_AR.unlink()
        print("[INFO] Shared memory cleaned up.")

if __name__ == "__main__":
    main()
