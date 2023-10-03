class Offset:

    def __init__(self, value: float, is_ratio: bool, x_shift: bool = False):
        self.value = value
        self.is_ratio = is_ratio
        self.x_shift = x_shift

    def get_offset(self, x):
        if self.is_ratio:
            return x * self.x_shift + x * self.value
        else:
            return x * self.x_shift + self.value