import cv2
import numpy as np
from numpy import ndarray

template_border = cv2.cvtColor(cv2.imread('sample_end_race_borders_single.png', 0), cv2.COLOR_BGRA2BGR)
template_press_a_button = cv2.cvtColor(cv2.imread('sample_end_Race_press_button_prompt.png', 0), cv2.COLOR_BGRA2BGR)


def check_if_black_bars_exist(image: ndarray):
    """
    Checks if black bars exist in the image.
    :param image: Image to check.
    :return: True if black bars exist, False otherwise.
    """
    global template_border

    width = template_border.shape[1]
    height = template_border.shape[0]

    res = cv2.matchTemplate(image, template_border, cv2.TM_CCOEFF_NORMED)
    threshold = .8
    loc = np.where(res >= threshold)
    for point in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 2)

    cv2.imshow('image', image)
    cv2.waitKey(1)
    cv2.destroyWindow('image')


def check_if_press_a_button_exists(image: ndarray):
    global template_press_a_button

    res = cv2.matchTemplate(image, template_press_a_button, cv2.TM_CCOEFF_NORMED)
    threshold = .7
    loc = np.where(res >= threshold)

    if len(loc[0]) > 0:
        return True  # Template is on the screen
    else:
        return False
