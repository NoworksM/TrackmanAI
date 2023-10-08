import asyncio
import json
import os
import threading
from datetime import datetime
from os import path
from queue import Queue

from input import TM2020OpenPlanetClient, Trackmania2020Data
from recording import ScreenRecorder
import time
from pynput import keyboard
import cv2
from numpy import ndarray

sleep_time = 1 / 20
data_base_path = 'C:\\Users\\Noworks\\Documents\\Trackmania\\TrainingData'

start_channel = Queue()


class FrameSnapshot:
    def __init__(self, run_start_time: float, frame_time: float, image: ndarray, vehicle_data: Trackmania2020Data):
        self.run_start_time: float = run_start_time
        self.frame_time: float = frame_time
        self.image: ndarray = image
        self.vehicle_data: Trackmania2020Data = vehicle_data


async def main():
    global start_channel
    screen_recorder = ScreenRecorder()
    openplanet_client = TM2020OpenPlanetClient()

    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    snapshot_channel = Queue()

    # Start saving thread
    save_thread = threading.Thread(target=save_queue, args=(snapshot_channel,), daemon=True)
    save_thread.start()

    while True:
        start_channel.get()
        await record_run(screen_recorder, openplanet_client, snapshot_channel)

    save_thread.cancel()


def on_press(key):
    try:
        if key.char == 'q':
            start_channel.put(True)
            print('Starting run')
    except:
        pass


async def record_run(screen_recorder: ScreenRecorder, openplanet_client: TM2020OpenPlanetClient,
                     channel: Queue):
    run_start_time = time.time_ns()
    vehicle_data = openplanet_client.retrieve_data()

    while not vehicle_data.terminated:
        try:
            frame_time = time.perf_counter()
            frame = screen_recorder.record_downsampled_frame(4)
            vehicle_data = openplanet_client.retrieve_data()

            current = time.time_ns()

            channel.put(FrameSnapshot(run_start_time, current, frame, vehicle_data))

            current_time = time.perf_counter()

            await asyncio.sleep(sleep_time - (current_time - frame_time))
        except:
            pass

    print('Run ended')


def save_queue(channel: Queue):
    while True:
        try:
            frame_snapshot = channel.get()

            run_datetime = datetime.fromtimestamp(frame_snapshot.run_start_time / 1e9)

            base_path = path.join(data_base_path, run_datetime.strftime('%Y-%m-%d_%H-%M-%S'),
                                  f'{frame_snapshot.frame_time % 1e12}')
            frame_path = base_path + '.bmp'
            data_path = base_path + '.json'

            # Create directories
            if not path.exists(path.dirname(frame_path)):
                os.makedirs(path.dirname(frame_path))

            # Write image to file
            cv2.imwrite(frame_path, frame_snapshot.image)
            print(f'Saving image at {datetime.now()}')

            # Write data to json file
            with open(data_path, 'w') as f:
                json.dump(frame_snapshot.vehicle_data.__dict__, f)
        except:
            pass


if __name__ == '__main__':
    asyncio.run(main())
