__all__ = ["YaqdPicotechAdcTriggered"]

import asyncio
import ctypes
import numpy as np

from typing import Dict, Any, List

from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import adc2mV, mV2adc, assert_pico2000_ok
from yaqd_core import Sensor


class YaqdPicotechAdcTriggered(Sensor):
    _kind = "yaqd-picotech-adc-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        # Perform any unique initialization

        self._channel_names = ["channel"]
        self._channel_units = {"channel": "units"}

        status["openUnit"] = ps.ps2000_open_unit()
        assert_pico2000_ok(status["openUnit"])
        self.chandle = ctypes.c_int16(status["openUnit"])

        status["setsig"] = ps.ps2000_set_sig_gen_built_in(
            chandle,
            0,
            ctypes.c_uint32(1000000),
            3,
            1000,
            1000,
            0,
            0,
            ctypes.c_int32(0),
            0
        )

    async def _measure(self):
        return {"channel": 0}
