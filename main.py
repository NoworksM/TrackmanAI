import ctypes

import cv2
import win32gui
import win32ui
import win32con
from time import perf_counter
import imgutils
from input import TrackmaniaInputStateLog

w = 2560  # set this
h = 1440  # set this

trackmania_window_handle = win32gui.FindWindow(None, 'Trackmania')

if trackmania_window_handle == 0:
    raise Exception('Trackmania Not Running')

inputStateLog = TrackmaniaInputStateLog(recording_frequency=30, log=True)
inputStateLog.start_input_logging()

img_pressed = cv2.imread('key_pressed.png', cv2.IMREAD_COLOR)
img_released = cv2.imread('key_released.png', cv2.IMREAD_COLOR)

count = 0
start = perf_counter()

while True:
    # windowDeviceContext = win32gui.GetWindowDC(trackmaniaWindowHandle)
    # deviceContext = win32ui.CreateDCFromHandle(windowDeviceContext)
    # compatibleDeviceContext = deviceContext.CreateCompatibleDC()
    # dataBitMap = win32ui.CreateBitmap()
    # dataBitMap.CreateCompatibleBitmap(deviceContext, w, h)
    # compatibleDeviceContext.SelectObject(dataBitMap)
    # compatibleDeviceContext.BitBlt((0, 0), (w, h), deviceContext, (0, 0), win32con.SRCCOPY)
    # Get window's device context

    hDC = win32gui.GetWindowDC(trackmania_window_handle)
    memory_device_context = win32ui.CreateDCFromHandle(hDC)

    # Create memory DC and bitmap
    compatible_memory_device_context = memory_device_context.CreateCompatibleDC()
    rect = win32gui.GetWindowRect(trackmania_window_handle)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    data_bitmap = win32ui.CreateBitmap()
    data_bitmap.CreateCompatibleBitmap(memory_device_context, width, height)
    compatible_memory_device_context.SelectObject(data_bitmap)

    # Copy window contents into bitmap
    compatible_memory_device_context.BitBlt((0, 0), (width, height), memory_device_context, (0, 0), win32con.SRCCOPY)

    grayscale_np_array = imgutils.downsample_cv2(data_bitmap, 4)

    cv2.putText(grayscale_np_array, f'{inputStateLog.state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                1)

    cv2.imshow('array', grayscale_np_array)

    # Free Resources
    memory_device_context.DeleteDC()
    compatible_memory_device_context.DeleteDC()
    # win32gui.ReleaseDC(trackmania_window_handle, windowDeviceContext)
    win32gui.DeleteObject(data_bitmap.GetHandle())

    count += 1

    if count % 100 == 0:
        current = perf_counter()

        print(
            f'Average capture: {count / (current - start)}\nAverage Frametime: {round((current - start) * 1000 / count, 2)}ms')

    # Handle windows messages
    win32gui.PumpWaitingMessages()

# Cleanup
win32gui.UnregisterClass(wndclass_atom, None)
