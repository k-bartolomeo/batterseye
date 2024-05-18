import time


def update_time(start_time: int) -> None:
    """Prints time since given start time"""
    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Time Elapsed: {mins}m {seconds}s')
    