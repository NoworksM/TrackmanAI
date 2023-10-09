import ctypes
import time

import pygetwindow as gw
import pyautogui

from input.keyboard_control import ReleaseKey, PressKey

VK_W = 0x11
VK_A = 0x1E
VK_S = 0x1F
VK_D = 0x20
VK_DELETE = 0xD3

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101


# Get the window handle
def get_window_handle(window_title):
    return ctypes.windll.user32.FindWindowW(0, window_title)


def focus_window(hwnd):
    ctypes.windll.user32.SetForegroundWindow(hwnd)


# Post the key press and release messages
def send_keypress(hwnd, key_code):
    ctypes.windll.user32.PostMessageW(hwnd, WM_KEYDOWN, key_code, 0)
    time.sleep(0.1)  # This is just to simulate a brief keypress. Adjust if necessary.
    ctypes.windll.user32.PostMessageW(hwnd, WM_KEYUP, key_code, 0)


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ki", KEYBDINPUT),
        ("padding", ctypes.c_ubyte * 8)
    ]


def press_key(key_code):
    extra = ctypes.c_ulong(0)
    ii_ = INPUT(type=INPUT_KEYBOARD,
                ki=KEYBDINPUT(wVk=key_code, wScan=0, dwFlags=0, time=0, dwExtraInfo=ctypes.pointer(extra)))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))


def release_key(key_code):
    extra = ctypes.c_ulong(0)
    ii_ = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=key_code, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0,
                                                   dwExtraInfo=ctypes.pointer(extra)))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))


def send_key_down(hwnd, key_code):
    ctypes.windll.user32.PostMessageW(hwnd, WM_KEYDOWN, key_code, 0)


def send_key_up(hwnd, key_code):
    ctypes.windll.user32.PostMessageW(hwnd, WM_KEYUP, key_code, 0)


class TrackmaniaKeyboardDriver:
    # init with pynput keyboard writer
    def __init__(self, window_name='Trackmania'):
        self.window_name = window_name
        self.is_w_pressed = False
        self.is_a_pressed = False
        self.is_s_pressed = False
        self.is_d_pressed = False
        self.trackmania_window = None
        self._get_trackmania_window()

    def _get_trackmania_window(self):
        # Get the Trackmania window
        self.trackmania_window = get_window_handle(self.window_name)

    def _focus_window(self):
        # Bring the Trackmania window to the front
        if self.trackmania_window:
            focus_window(self.trackmania_window)
        else:
            self._get_trackmania_window()
            if self.trackmania_window:
                focus_window(self.trackmania_window)
            else:
                raise Exception("Failed to focus on the Trackmania window!")

    def reset(self):
        self._focus_window()
        if self.is_w_pressed:
            PressKey(VK_W)
            self.is_w_pressed = False
        if self.is_a_pressed:
            PressKey(VK_A)
            self.is_a_pressed = False
        if self.is_s_pressed:
            PressKey(VK_S)
            self.is_s_pressed = False
        if self.is_d_pressed:
            PressKey(VK_D)
            self.is_d_pressed = False

    def accelerate(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_W)
        self.is_w_pressed = True

    def brake(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_S)
        self.is_s_pressed = True

    def turn_left(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_A)
        self.is_a_pressed = True

    def turn_right(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_D)
        self.is_d_pressed = True

    def accelerate_left(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_W)
        self.is_w_pressed = True
        ReleaseKey(VK_A)
        self.is_a_pressed = True

    def accelerate_right(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_W)
        self.is_w_pressed = True
        ReleaseKey(VK_D)
        self.is_d_pressed = True

    def brake_left(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_S)
        self.is_s_pressed = True
        ReleaseKey(VK_A)
        self.is_a_pressed = True

    def brake_right(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_S)
        self.is_s_pressed = True
        ReleaseKey(VK_D)
        self.is_d_pressed = True

    def drift_left(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_W)
        self.is_w_pressed = True
        ReleaseKey(VK_S)
        self.is_s_pressed = True
        ReleaseKey(VK_A)
        self.is_a_pressed = True

    def drift_right(self):
        self._focus_window()
        self.reset()
        ReleaseKey(VK_W)
        self.is_w_pressed = True
        ReleaseKey(VK_S)
        self.is_s_pressed = True
        ReleaseKey(VK_D)
        self.is_d_pressed = True

    def restart(self):
        self.reset()
        self._focus_window()
        PressKey(VK_DELETE)
        time.sleep(0.1)
        ReleaseKey(VK_DELETE)
