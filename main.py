import multiprocessing
from multiprocessing import shared_memory
from image_process import pi_cam_main
from hud_display import hud_main

def create_shared_memory(name, size):
    """Helper function to create or attach to an existing shared memory block."""
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        return shared_memory.SharedMemory(name=name, create=False)

def cleanup_shared_memory(*shared_memories):
    """Helper function to clean up shared memory blocks."""
    for shm in shared_memories:
        shm.close()
        shm.unlink()

def main():
    # Queues for inter-process communication
    queue_sign_ids = multiprocessing.Queue()
    queue_info = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    # Create shared memory blocks
    shm_vid = create_shared_memory(name="video_frame", size=720 * 1280 * 3)  # 1080*1920*3
    shm_AR = create_shared_memory(name="ar_info", size=288)

    # Define processes
    process_QR = multiprocessing.Process(target=pi_cam_main, args=(queue_sign_ids, queue_info, stop_event))
    process_HUD = multiprocessing.Process(target=hud_main, args=(queue_sign_ids, queue_info, stop_event))

    try:
        # Start and join processes
        process_QR.start()
        process_HUD.start()
        process_QR.join()
        process_HUD.join()
    except KeyboardInterrupt:
        # Handle graceful shutdown on interrupt
        stop_event.set()
        process_QR.join()
        process_HUD.join()
    finally:
        # Clean up shared memory
        cleanup_shared_memory(shm_vid, shm_AR)
        print("[INFO] Shared memory cleaned up.")

if __name__ == "__main__":
    main()
