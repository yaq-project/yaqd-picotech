from enum import Enum
import ctypes

# maximum ADC count value
# drivers normalize to 16 bit (15 bit signed) regardless of resolution
__maxADC__ = ctypes.c_uint16(2**15)


class Waveform(Enum):
    SINE = 0
    SQUARE = 1
    TRIANGLE = 2
    RAMP_UP = 3
    RAMP_DOWN = 4
    DC = 5


class ChannelName(Enum):
    A = 0
    B = 1
    C = 2
    D = 3


class Coupling(Enum):
    DC = True
    AC = False


class ChannelRange(Enum):
    # connect name to range code and half-range
    # mV_10 = (0, 10)  # not valid for 2204
    mV_20 = (1, 20)
    mV_50 = (2, 50)
    mV_100 = (3, 100)
    mV_200 = (4, 200)
    mV_500 = (5, 500)
    V_1 = (6, 1000)
    V_2 = (7, 2000)
    V_5 = (8, 5000)
    V_10 = (9, 10_000)
    V_20 = (10, 20_000)
