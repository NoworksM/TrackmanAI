from enum import Enum


class Action(Enum):
    NOTHING = 0
    FORWARDS = 1
    BACKWARDS = 2
    LEFT = 3
    RIGHT = 4
    FORWARDS_LEFT = 5
    FORWARDS_RIGHT = 6
    BACKWARDS_LEFT = 7
    BACKWARDS_RIGHT = 8
    DRIFT_LEFT = 9
    DRIFT_RIGHT = 10