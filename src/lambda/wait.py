import sys
import time


def wait(seconds, tick=12):
    """
    Waits for a specified number of seconds, while also displaying an animated
    spinner.

    :param seconds: The number of seconds to wait.
    :param tick: The number of frames per second used to animate the spinner.
    """
    progress = '|/-\\'
    waited = 0
    while waited < seconds:
        for frame in range(tick):
            sys.stdout.write(f"\r{progress[frame % len(progress)]}")
            sys.stdout.flush()
            time.sleep(1/tick)
        waited += 1
    sys.stdout.write("\r")
    sys.stdout.flush()
