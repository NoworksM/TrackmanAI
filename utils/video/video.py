import subprocess


def convert_images_to_video(image_pattern: str, output_video: str, framerate: float):
    """
    Convert a sequence of images to a video using FFmpeg.

    Parameters:
    - image_pattern: A pattern for image filenames, e.g., "img%03d.jpg" for img001.jpg, img002.jpg, etc.
    - output_video: Name of the output video file.
    - framerate: Frame rate of the resulting video.

    Returns:
    - None
    """

    # FFmpeg command
    cmd = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', image_pattern,
        '-c:v', 'libx264',  # Codec to use, you can change this as needed
        '-pix_fmt', 'yuv420p',  # Pixel format, usually needed for compatibility reasons
        '-b:v', '10M',
        output_video
    ]

    subprocess.run(cmd)
