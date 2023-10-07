# Define BITMAPINFO structure (relevant parts for grayscale)
import ctypes


class BitmapInfo(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_long),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_short),
        ("biBitCount", ctypes.c_short),
        ("biCompression", ctypes.c_long),
        # ... (other members can be added for a complete definition)
    ]