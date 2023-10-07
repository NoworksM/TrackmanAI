import numpy as np
import numexpr as ne
import win32gui
import win32ui
import cv2


def win32_bitmap_to_numpy_gray(bitmap, downsampling_factor=2):
    """
    Convert a Win32 bitmap to a grayscale numpy array.

    Parameters:
        - bitmap: The Win32 bitmap object

    Returns:
        - A grayscale numpy array
    """

    # Get bitmap's dimensions
    bmpinfo = bitmap.GetInfo()
    width, height = bmpinfo['bmWidth'], bmpinfo['bmHeight']

    # Get bitmap data
    bmpstr = bitmap.GetBitmapBits(True)

    # Create numpy array from the bitmap data
    img = np.fromstring(bmpstr, dtype=np.uint8)
    img.shape = (height, width, 4)

    # Convert to grayscale using the standard formula
    # gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    # gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    gray_img = ne.evaluate("0.299*red + 0.587*green + 0.114*blue")

    downsampled_width = int(width / downsampling_factor)
    downsampled_height = int(height / downsampling_factor)

    downsampled = gray_img.reshape((height, downsampled_width, downsampling_factor, 1))
    downsampled = np.mean(downsampled, axis=2)

    downsampled = downsampled.reshape((downsampled_height, downsampling_factor, downsampled_width, 1))
    downsampled = np.mean(downsampled, axis=1)

    # downsampled = np.zeros((downsampled_height, downsampled_width), dtype=img.dtype)
    #
    # for i in range(downsampling_factor):
    #     for j in range(downsampling_factor):
    #         downsampled += img[i::downsampling_factor, j::downsampling_factor, :]
    #
    # downsampled = ne.evaluate("downsampled / (factor * factor)")

    # Return the grayscale image
    return downsampled.astype(np.uint8)


def downsample_cv2(bitmap, factor=2):
    bmpinfo = bitmap.GetInfo()
    width, height = bmpinfo['bmWidth'], bmpinfo['bmHeight']

    # Get bitmap data
    bmpstr = bitmap.GetBitmapBits(True)

    # Create numpy array from the bitmap data
    img = np.fromstring(bmpstr, dtype=np.uint8)
    img.shape = (height, width, 4)

    downsampled = cv2.resize(img, (int(img.shape[1] / factor), int(img.shape[0] / factor)))
    downsampled = cv2.cvtColor(downsampled, cv2.COLOR_BGR2GRAY)

    return downsampled.astype(np.uint8)
