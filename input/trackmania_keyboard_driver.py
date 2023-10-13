import ctypes
import time

from input.keyboard_control import ReleaseKey, PressKey
from .trackmania_action import TrackmaniaAction

VK_W = 0x11
VK_A = 0x1E
VK_S = 0x1F
VK_D = 0x20
VK_SPACE = 0x39
VK_DOWN = 0xD0
VK_ENTER = 0x1C
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
        self.current_action = TrackmaniaAction.Nothing
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

    def perform_action(self, action):
        was_accelerating = TrackmaniaAction.has_flag(self.current_action, TrackmaniaAction.Accelerate)
        was_braking = TrackmaniaAction.has_flag(self.current_action, TrackmaniaAction.Brake)
        was_turning_left = TrackmaniaAction.has_flag(self.current_action, TrackmaniaAction.Left)
        was_turning_right = TrackmaniaAction.has_flag(self.current_action, TrackmaniaAction.Right)

        is_accelerating = TrackmaniaAction.has_flag(action, TrackmaniaAction.Accelerate)
        is_braking = TrackmaniaAction.has_flag(action, TrackmaniaAction.Brake)
        is_turning_left = TrackmaniaAction.has_flag(action, TrackmaniaAction.Left)
        is_turning_right = TrackmaniaAction.has_flag(action, TrackmaniaAction.Right)

        if was_accelerating and not is_accelerating:
            self._focus_window()
            ReleaseKey(VK_W)
        elif not was_accelerating and is_accelerating:
            self._focus_window()
            PressKey(VK_W)
        if was_braking and not is_braking:
            self._focus_window()
            ReleaseKey(VK_S)
        elif not was_braking and is_braking:
            self._focus_window()
            PressKey(VK_S)
        if was_turning_left and not is_turning_left:
            self._focus_window()
            ReleaseKey(VK_A)
        elif not was_turning_left and is_turning_left:
            self._focus_window()
            PressKey(VK_A)
        if was_turning_right and not is_turning_right:
            self._focus_window()
            ReleaseKey(VK_D)
        elif not was_turning_right and is_turning_right:
            self._focus_window()
            PressKey(VK_D)

        self.current_action = action

    def reset(self):
        self._focus_window()
        if self.current_action.has_flag(TrackmaniaAction.Accelerate):
            ReleaseKey(VK_W)
        if self.current_action.has_flag(TrackmaniaAction.Brake):
            ReleaseKey(VK_A)
        if self.current_action.has_flag(TrackmaniaAction.Left):
            ReleaseKey(VK_S)
        if self.current_action.has_flag(TrackmaniaAction.Right):
            ReleaseKey(VK_D)

        self.current_action = TrackmaniaAction.Nothing

    def accelerate(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.Accelerate)

    def brake(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.Brake)

    def turn_left(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.Left)

    def turn_right(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.Right)

    def accelerate_left(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.AccelerateLeft)

    def accelerate_right(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.AccelerateRight)

    def brake_left(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.BrakeLeft)

    def brake_right(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.BrakeRight)

    def drift_left(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.DriftLeft)

    def drift_right(self):
        self._focus_window()
        self.perform_action(TrackmaniaAction.DriftRight)

    def start_course_again(self):
        time.sleep(8)
        self._focus_window()
        PressKey(VK_ENTER)
        time.sleep(0.25)
        self._focus_window()
        ReleaseKey(VK_ENTER)

    def save_replay_and_start_course_again(self):
        time.sleep(8)
        self._focus_window()
        PressKey(VK_DOWN)
        time.sleep(0.25)
        self._focus_window()
        ReleaseKey(VK_DOWN)
        time.sleep(3)
        self._focus_window()
        PressKey(VK_ENTER)
        time.sleep(0.25)
        self._focus_window()
        ReleaseKey(VK_ENTER)
        time.sleep(3)
        self._focus_window()
        PressKey(VK_ENTER)
        time.sleep(0.25)
        self._focus_window()
        ReleaseKey(VK_ENTER)

    def restart(self, frame=None):
        self.reset()
        self._focus_window()
        # if frame is not None and detection.check_if_press_a_button_exists(frame):
        #     self._focus_window()
        #     PressKey(VK_SPACE)
        #     time.sleep(0.25)
        #     self._focus_window()
        #     ReleaseKey(VK_SPACE)
        #     self.save_replay_and_start_course_again()
        # else:
        #     PressKey(VK_DELETE)
        #     time.sleep(0.25)
        #     self._focus_window()
        #     ReleaseKey(VK_DELETE)

        PressKey(VK_DELETE)
        time.sleep(0.25)
        self._focus_window()
        ReleaseKey(VK_DELETE)
