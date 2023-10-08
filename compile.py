import json
import math
import os

from PIL import Image

import config
import utils.movement as movement
import utils.video as video


def main():
    # Get all folders in the training data directory
    folders = os.listdir(config.data_base_path)

    for folder in folders:
        compile_folder(os.path.join(config.data_base_path, folder))


def rename_timestamp_files_to_frame_number(path: str):
    files = os.listdir(path)

    bitmap_files = [file for file in files if file.endswith('.bmp') or file.endswith('.png')]
    bitmap_files.sort(key=lambda x: int(x.split('.')[0]))

    for i in range(len(bitmap_files)):
        bitmap_file = bitmap_files[i]
        json_file = bitmap_file[:-3] + 'json'

        os.rename(os.path.join(path, bitmap_file), os.path.join(path, f'{i:0{config.frame_naming_places}}.bmp'))
        os.rename(os.path.join(path, json_file), os.path.join(path, f'{i:0{config.frame_naming_places}}.json'))


def compile_folder(path: str):
    files = os.listdir(path)

    if 'route.json' in files or 'route_video.mp4' in files:
        route_data_path = os.path.join(path, 'route.json')
        video_path = os.path.join(path, 'route_video.mp4')

        if os.path.exists(route_data_path) and not os.path.exists(video_path) or not os.path.exists(
                route_data_path) and os.path.exists(video_path):
            if os.path.exists(route_data_path):
                os.remove(route_data_path)
            if os.path.exists(video_path):
                os.remove(video_path)
        # Ensure files have data and ensure json file is not an empty array
        elif os.path.getsize(route_data_path) == 0 or os.path.getsize(video_path) == 0:
            if os.path.exists(route_data_path):
                os.remove(route_data_path)
            if os.path.exists(video_path):
                os.remove(video_path)
        elif os.path.getsize(route_data_path) > 0:
            with open(route_data_path, 'r') as f:
                route_data = json.load(f)

                if len(route_data) == 0:
                    try:
                        if os.path.exists(route_data_path):
                            os.remove(route_data_path)
                        if os.path.exists(video_path):
                            os.remove(video_path)
                    except:
                        return
                else:
                    return
        else:
            return

    assert not os.path.exists(os.path.join(path, 'route.json'))
    assert not os.path.exists(os.path.join(path, 'route_video.mp4'))

    route = []

    bitmap_files = [file for file in files if file.endswith('.bmp')]
    bitmap_files.sort(key=lambda x: int(x.split('.')[0]))

    if len(bitmap_files) == 0:
        return

    # Check if files start at 1
    if int(bitmap_files[0].split('.')[0]) != '0' * config.frame_naming_places:
        rename_timestamp_files_to_frame_number(path)
        bitmap_files = [file for file in files if file.endswith('.bmp')]
        bitmap_files.sort(key=lambda x: int(x.split('.')[0]))

    for bitmap_file in bitmap_files:
        json_file = os.path.join(path, bitmap_file[:-3] + 'json')

        if not os.path.exists(json_file):
            raise f'paired JSON file {json_file} does not exist'

        with open(json_file, 'r') as f:
            frame_data = json.load(f)

            route.append(frame_data)

    reward_distance = 0.0

    check_frames = config.polling_rate * config.reward_timeframe_seconds

    for i in range(check_frames):
        if i >= len(route):
            break

        # calculate distance between this frame and the next from x, y, z coordinates
        if i + 1 < len(route):
            reward_distance += movement.calculate_distance(route[i]['x'], route[i]['y'], route[i]['z'],
                                                           route[i + 1]['x'], route[i + 1]['y'], route[i + 1]['z'])

    for i in range(len(route)):
        frame_data = route[i]

        frame_data['reward'] = reward_distance

        if i + check_frames > len(route):
            overlap = i + check_frames - len(route)

            frame_data['reward'] += 5 * math.pow(100, overlap / check_frames)
        elif i + check_frames + 1 < len(route):
            reward_distance -= movement.calculate_distance(route[i]['x'], route[i]['y'], route[i]['z'],
                                                           route[i + 1]['x'], route[i + 1]['y'], route[i + 1]['z'])
            reward_distance += movement.calculate_distance(route[i + check_frames]['x'], route[i + check_frames]['y'],
                                                           route[i + check_frames]['z'],
                                                           route[i + check_frames + 1]['x'],
                                                           route[i + check_frames + 1]['y'],
                                                           route[i + check_frames + 1]['z'])

    assert len(bitmap_files) == len(route)

    with open(os.path.join(path, 'route.json'), 'w') as f:
        json.dump(route, f, indent=4)

    video_path = os.path.join(path, 'route_video.mp4')
    images_path = os.path.join(path, f'%0{config.frame_naming_places}d.bmp')

    video.convert_images_to_video(images_path, video_path, config.polling_rate)


def convert_bitmap_to_png(path: str):
    # Get png name from path
    png_path = path[:-3] + 'png'

    bitmap = Image.open(path)
    bitmap.save(png_path, "png")
    bitmap.close()

    # Delete bitmap
    os.remove(path)

    return png_path


if __name__ == '__main__':
    main()
