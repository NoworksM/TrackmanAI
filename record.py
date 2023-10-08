import asyncio
import json
import os
from os import path

from input import TM2020OpenPlanetClient, Trackmania2020Data
from recording import ScreenRecorder
from time import perf_counter
from pynput import keyboard
import cv2
from numpy import ndarray

sleep_time = 1 / 20
data_base_path = 'C:\\Users\\Noworks\\Documents\\Trackmania\\TrainingData'

start_channel = asyncio.Queue[None]()


class FrameSnapshot:
    def __init__(self, run_start_time: float, frame_time: float, image: ndarray, vehicle_data: Trackmania2020Data):
        self.run_start_time: float = run_start_time
        self.frame_time: float = frame_time
        self.image: ndarray = image
        self.vehicle_data: Trackmania2020Data = vehicle_data


async def main():
    screen_recorder = ScreenRecorder()
    openplanet_client = TM2020OpenPlanetClient()

    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    snapshot_channel: asyncio.Queue[FrameSnapshot] = asyncio.Queue()

    save_thread = asyncio.create_task(save_queue(snapshot_channel))

    while True:
        await start_channel.get()
        await record_run(screen_recorder, openplanet_client, snapshot_channel)

    save_thread.cancel()


def on_press(key):
    if key.char == 'q':
        start_channel.put(None)
        print('Starting run')


async def record_run(screen_recorder: ScreenRecorder, openplanet_client: TM2020OpenPlanetClient,
                     channel: asyncio.Queue[FrameSnapshot]):
    run_start_time = perf_counter()
    vehicle_data = openplanet_client.retrieve_data()

    while (not vehicle_data.terminated) or True:
        try:
            frame_time = perf_counter()
            bitmap = screen_recorder.record_frame_bitmap()
            vehicle_data = openplanet_client.retrieve_data()

            await channel.put(FrameSnapshot(run_start_time, frame_time, bitmap, vehicle_data))

            current_time = perf_counter()

            await asyncio.sleep(sleep_time - (current_time - frame_time))
        except:
            pass

    print('Run ended')


async def save_queue(channel: asyncio.Queue[FrameSnapshot]):
    while True:
        frame_snapshot = await channel.get()

        base_path = path.join(data_base_path, f'{round(frame_snapshot.run_start_time)}',
                              f'{round(frame_snapshot.frame_time)}')
        frame_path = base_path + '.bmp'
        data_path = base_path + '.json'

        # Create directories
        if not path.exists(path.dirname(frame_path)):
            os.makedirs(path.dirname(frame_path))

        # Write image to file
        cv2.imwrite(frame_path, frame_snapshot.image)

        # Write data to json file
        with open(data_path, 'w') as f:
            json.dump(frame_snapshot.vehicle_data, f)


if __name__ == '__main__':
    asyncio.run(main())
