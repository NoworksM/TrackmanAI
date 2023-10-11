from imgutils import detection
from recording import ScreenRecorder

screen_recorder = ScreenRecorder()

while True:
    frame = screen_recorder.record_downsampled_frame(4)
    detection.check_if_black_bars_exist(frame)
    detection.check_if_press_a_button_exists(frame)