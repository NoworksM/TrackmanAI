from enum import Enum


class TrackmaniaAction(Enum):
    Nothing = 0
    Accelerate = 1
    Brake = 2
    Drift = 1 | 2
    Left = 4
    Right = 8
    AccelerateLeft = 1 | 4
    AccelerateRight = 1 | 8
    BrakeLeft = 2 | 4
    BrakeRight = 2 | 8
    DriftLeft = 1 | 2 | 4
    DriftRight = 1 | 2 | 8

    def has_flag(self, flag):
        return self.value & flag.value == flag.value

    def has_flags(self, *args):
        for flag in args:
            if not self.has_flag(flag):
                return False
        return True

    def has_xor_flags(self, first, second):
        return self.has_flag(first) ^ self.has_flag(second)
