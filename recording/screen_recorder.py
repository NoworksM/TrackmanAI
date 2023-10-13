import cv2
import numpy as np
import win32con
import win32gui
import win32ui


class ScreenRecorder:
    def __init__(self, window_name='Trackmania'):
        self._window_name = window_name
        self._trackmania_window_handle = win32gui.FindWindow(None, self._window_name)

    def record_frame_bitmap(self):
        hDC = win32gui.GetWindowDC(self._trackmania_window_handle)
        memory_device_context = win32ui.CreateDCFromHandle(hDC)

        # Create memory DC and bitmap
        compatible_memory_device_context = memory_device_context.CreateCompatibleDC()
        rect = win32gui.GetWindowRect(self._trackmania_window_handle)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        data_bitmap = win32ui.CreateBitmap()
        data_bitmap.CreateCompatibleBitmap(memory_device_context, width, height)
        compatible_memory_device_context.SelectObject(data_bitmap)

        # Copy window contents into bitmap
        compatible_memory_device_context.BitBlt((0, 0), (width, height), memory_device_context, (0, 0),
                                                win32con.SRCCOPY)

        return data_bitmap

    def record_frame_to_ndarray(self):
        hDC = win32gui.GetWindowDC(self._trackmania_window_handle)
        memory_device_context = win32ui.CreateDCFromHandle(hDC)

        # Create memory DC and bitmap
        compatible_memory_device_context = memory_device_context.CreateCompatibleDC()
        rect = win32gui.GetWindowRect(self._trackmania_window_handle)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        data_bitmap = win32ui.CreateBitmap()
        data_bitmap.CreateCompatibleBitmap(memory_device_context, width, height)
        old_bitmap = compatible_memory_device_context.SelectObject(data_bitmap)

        # Copy window contents into bitmap
        compatible_memory_device_context.BitBlt((0, 0), (width, height), memory_device_context, (0, 0),
                                                win32con.SRCCOPY)

        img = ScreenRecorder.bitmap_to_numpy_grayscale(data_bitmap)

        # Cleanup
        # De-select the bitmap
        compatible_memory_device_context.SelectObject(old_bitmap)
        # Delete the created resources
        win32gui.DeleteObject(data_bitmap.GetHandle())
        compatible_memory_device_context.DeleteDC()
        memory_device_context.DeleteDC()
        win32gui.ReleaseDC(self._trackmania_window_handle, hDC)

        return img

    def record_downsampled_frame(self, factor=2):
        return ScreenRecorder.downsample_image(self.record_frame_to_ndarray(), factor)

    @staticmethod
    def bitmap_to_numpy(bitmap):
        """Convert a windows Bitmap to a numpy array"""
        bitmap_info = bitmap.GetInfo()
        width, height = bitmap_info['bmWidth'], bitmap_info['bmHeight']

        # Get bitmap data
        bitmap_string = bitmap.GetBitmapBits(True)

        # Create numpy array from the bitmap data
        img = np.fromstring(bitmap_string, dtype=np.uint8)
        img.shape = (height, width, 4)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    def bitmap_to_numpy_grayscale(bitmap):
        """Convert a windows Bitmap to a numpy array"""
        bitmap_info = bitmap.GetInfo()
        width, height = bitmap_info['bmWidth'], bitmap_info['bmHeight']

        # Get bitmap data
        bitmap_string = bitmap.GetBitmapBits(True)

        # Create numpy array from the bitmap data
        img = np.fromstring(bitmap_string, dtype=np.uint8)
        img.shape = (height, width, 4)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    @staticmethod
    def downsample_image(image, factor=2):
        """Downsample a numpy image by a given factor"""
        downsampled = cv2.resize(image, (int(image.shape[1] / factor), int(image.shape[0] / factor)))

        if len(image.shape) > 2:
            downsampled.shape = (downsampled.shape[0], downsampled.shape[1], image.shape[2])
        else:
            downsampled.shape = (downsampled.shape[0], downsampled.shape[1])

        return downsampled.astype(np.uint8)

    @staticmethod
    def bgra_to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
