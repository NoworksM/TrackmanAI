import time
import threading
from pynput import keyboard

key_states = {
    "accelerate": False,
    "left": False,
    "right": False,
    "brake": False
}


class TrackmaniaInputStateLog:
    def __init__(self, recording_frequency, log=False):
        self.recording_frequency = recording_frequency
        self.input_states = []
        self._thread = None
        self._listener = None
        self._listening = False
        self.log = log
        self.state = key_states

    def add_frame(self, keyboard_state: dict):
        immutable_state = keyboard_state.copy()
        self.input_states.append(immutable_state)
        self.state = immutable_state

    def _record_inputs(self):
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        while self._listening:
            time.sleep(1 / self.recording_frequency)

    def start_input_logging(self):
        if not self._listening and self._thread is None:
            self._listening = True
            self._thread = threading.Thread(target=self._record_inputs)
            self._thread.daemon = True  # Setting this ensures the thread will exit when the main program exits
            self._thread.start()

    def stop_input_logging(self):
        if self._listening and self._thread is not None:
            self._listening = False
            self._thread.join()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.space:
                key_states["brake"] = True
                if self.log:
                    print('space pressed')
            elif key == keyboard.Key.up:
                key_states["accelerate"] = True
                if self.log:
                    print('up pressed')
            elif key == keyboard.Key.left:
                key_states["left"] = True
                if self.log:
                    print('left pressed')
            elif key == keyboard.Key.right:
                key_states["right"] = True
                if self.log:
                    print('right pressed')
            elif key == keyboard.Key.down:
                key_states["brake"] = True
                if self.log:
                    print('down pressed')
            elif key.char == 'w':
                key_states["accelerate"] = True
                if self.log:
                    print('w pressed')
            elif key.char == 'a':
                key_states["left"] = True
                if self.log:
                    print('a pressed')
            elif key.char == 's':
                key_states["brake"] = True
                if self.log:
                    print('s pressed')
            elif key.char == 'd':
                key_states["right"] = True
                if self.log:
                    print('d pressed')
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            if key == keyboard.Key.space:
                key_states["brake"] = False
                if self.log:
                    print('space released')
            elif key == keyboard.Key.up:
                key_states["accelerate"] = False
                if self.log:
                    print('up released')
            elif key == keyboard.Key.left:
                key_states["left"] = False
                if self.log:
                    print('left released')
            elif key == keyboard.Key.right:
                key_states["right"] = False
                if self.log:
                    print('right released')
            elif key == keyboard.Key.down:
                key_states["brake"] = False
                if self.log:
                    print('down released')
            elif key.char == 'w':
                key_states["accelerate"] = False
                if self.log:
                    print('w released')
            elif key.char == 'a':
                key_states["left"] = False
                if self.log:
                    print('a released')
            elif key.char == 's':
                key_states["brake"] = False
                if self.log:
                    print('s released')
            elif key.char == 'd':
                key_states["right"] = False
                if self.log:
                    print('d released')
        except AttributeError:
            pass
