import asyncio
import json
import os
import threading
import time
from datetime import datetime
from os import path
from queue import Queue

import cv2
from numpy import ndarray
from pynput import keyboard

import config
from input import TM2020OpenPlanetClient, Trackmania2020Data
from recording import ScreenRecorder

recording = False

start_channel = Queue()
delete_run_channel = Queue()


class FrameSnapshot:
    def __init__(self, frame_number: int, run_start_time: float, frame_time: float, image: ndarray,
                 vehicle_data: Trackmania2020Data):
        self.frame_number: int = frame_number
        self.run_start_time: float = run_start_time
        self.frame_time: float = frame_time
        self.image: ndarray = image
        self.vehicle_data: Trackmania2020Data = vehicle_data


async def main():
    global start_channel
    screen_recorder = ScreenRecorder()
    openplanet_client = TM2020OpenPlanetClient()

    keyboard_listener = keyboard.Listener(on_release=on_release)
    keyboard_listener.start()

    snapshot_channel = Queue()

    # Start saving thread
    save_thread = threading.Thread(target=save_queue, args=(snapshot_channel,), daemon=True)
    save_thread.start()

    remove_thread = threading.Thread(target=delete_run_queue, args=(delete_run_channel,), daemon=True)
    remove_thread.start()

    while True:
        start_channel.get()
        await record_run(screen_recorder, openplanet_client, snapshot_channel)

    save_thread.cancel()


def on_release(key):
    global recording
    try:
        if key.char == 'q':
            if recording:
                recording = False
                print('Cancelling run')
            else:
                recording = True
                start_channel.put(True)
                print('Starting run')
    except:
        pass


async def record_run(screen_recorder: ScreenRecorder, openplanet_client: TM2020OpenPlanetClient,
                     channel: Queue):
    global recording
    run_start_time_ns = time.time_ns()
    vehicle_data = openplanet_client.get_data()

    frame_number = 0

    while not vehicle_data.terminated and recording:
        try:
            frame_time = time.perf_counter()
            frame = screen_recorder.record_downsampled_frame(4)
            vehicle_data = openplanet_client.get_data()

            current = time.time_ns()

            channel.put(FrameSnapshot(frame_number, run_start_time_ns, current, frame, vehicle_data))

            current_time = time.perf_counter()

            frame_number += 1

            await asyncio.sleep(config.sleep_time - (current_time - frame_time))
        except:
            pass

    if recording:
        recording = False
        print('Run completed')
    else:
        print('Run cancelled')
        delete_run_channel.put(get_run_path(datetime.fromtimestamp(run_start_time_ns / 1e9)))


def delete_run_queue(channel: Queue):
    while True:
        try:
            run_dir = channel.get()

            if not path.exists(run_dir):
                return

            time.sleep(2.5)

            files = os.listdir(run_dir)

            for file in files:
                os.remove(path.join(run_dir, file))

            os.rmdir(run_dir)
        except:
            pass


def get_run_path(run_datetime: datetime) -> str:
    return path.join(config.data_base_path, run_datetime.strftime('%Y-%m-%d_%H-%M-%S'))


def save_queue(channel: Queue):
    while True:
        try:
            frame_snapshot = channel.get()

            run_datetime = datetime.fromtimestamp(frame_snapshot.run_start_time / 1e9)

            base_path = path.join(get_run_path(run_datetime),
                                  f'{frame_snapshot.frame_number:0{config.frame_naming_places}}')
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


def delete_run(timestamp: int):
    """Remove a cancelled run from the file system"""
    run_datetime = datetime.fromtimestamp(timestamp / 1e9)

    base_path = path.join(config.data_base_path, run_datetime.strftime('%Y-%m-%d_%H-%M-%S'))

    os.rmdir(base_path)


if __name__ == '__main__':
    asyncio.run(main())
