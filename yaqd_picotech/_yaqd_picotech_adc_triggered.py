__all__ = ["YaqdPicotechAdcTriggered"]

import asyncio
import ctypes
import numpy as np

from picosdk.functions import adc2mV, mV2adc
from typing import Dict, Any, List
from yaqd_core import Sensor

# todo: parse range codes based on psx000.PSx000_VOLTAGE_RANGE dict
# ranges = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
# code_to_range = {i+1: ranges[i] for i in range(len(ranges))}
range_to_code = {
    "20 mV": 1,
    "50 mV": 2,
    "100 mV": 3,
    "200 mV": 4,
    "500 mV": 5,
    "1 V": 6,
    "2 V": 7,
    "5 V": 8,
    "10 V": 9,
    "20 V": 10
}

wave_type_to_code = {
    k: i for i, k in enumerate([
        "sine", "square", "triangle", "ramp_up", "ramp_down", "dc"
        ]
    )
}

# maximum ADC count value
# drivers normalize to 16 bit (15 bit signed) regardless of resolution
maxADC = ctypes.c_uint16(2**15)

# ddk: ignore chopper class for now; only two physical channels to test


@dataclass
class Channel:
    name: str
    range: str = "1 V"
    physical_channel: int
    enabled: bool = True
    coupling: str = "DC"
    invert: bool = False
    use_baseline: bool = False
    baseline_start: int
    baseline_stop: int
    baseline_presample: int = 0
    baseline_method: str
    signal_presample: int = 0

    def volts_to_adc(self, x):
        return mV2adc(x / 1e3, range_to_code[self.range], maxADC)

    def adc_to_volts(self, x):
        return adc2mV(x, range_to_code[self.range], maxADC) / 1e3


class YaqdPicotechAdcTriggered(Sensor):
    _kind = "yaqd-picotech-adc-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)

        self._channels = []
        for k, d in self._config["channels"].items():
            channel = Channel(**d, physical_channel=k)
            self._channels.append(channel)
        self._channel_names = [c.name for c in self._channels if c.enabled]  # expected by parent
        # todo: readout is in mV currently, so adjust accordingly
        self._channel_units = {k: "V" for k in self._channel_names}  # expected by parent
        self.timeInterval = ctypes.c_int32()
        self.timeUnits = ctypes.c_int32()

        # check that all physical channels are unique
        x = []
        x += [c.physical_channel for c in self._channels]
        assert len(set(x)) == len(x)

        assert _config.model == "ps2000"

        # finish
        self._open_unit()
        self._set_channels()
        self._set_block_time()
        # trigger
        if self._config.is_self_triggered:
            self._set_awg_trigger()
        self._set_trigger()

    def _open_unit(self):
        from picosdk.ps2000 import ps2000
        from picosdk.functions import assert_pico2000_ok

        status = ps2000.ps2000_open_unit()
        assert_pico2000_ok(status)
        self.chandle = ctypes.c_int16(status)

    def _set_channels(self):
        from picosdk.ps2000 import ps2000
        from picosdk.functions import assert_pico2000_ok

        for c in self._channels:
            status = ps2000.ps2000_set_channel(
                self.chandle,
                c.physical_channel,  # channel
                c.enabled,  # enabled
                c.coupling,  # dc (True) / ac (False)
                range_to_code[c.range],  # range code
            )
            assert_pico2000_ok(status)

    def _set_trigger(self):
        from picosdk.ps2000 import ps2000
        from picosdk.functions import assert_pico2000_ok
        trigger_channel = [c for c in self.channels if c.name==self.config.trigger_source][0]
        status = ps2000.ps2000_set_trigger(
            self.chandle,
            trigger_channel.physical_channel,  # todo: convert to physical channel
            trigger_channel.V_to_adc(0),  # threshold
            0,  # direction (0=rising, 1=falling)
            -50, # delay the delay, as a percentage of the requested number of data points, between
            #      the trigger event and the start of the block
            5  # ms to wait before collecting if no trigger recieved (0 = infinity)
        )
        assert_pico2000_ok(status)

    def _set_awg_trigger(self):
        from picosdk.ps2000 import ps2000
        from picosdk.functions import assert_pico2000_ok
        # awg
        status = ps2000.ps2000_set_sig_gen_built_in(
            self.chandle,
            0,  # offset voltage (uV)
            ctypes.c_uint32(1000000),  # peak-to-peak voltage (uV)
            wave_type_to_code["square"],  # wavetype code
            1000,  # start frequency (Hz)
            1000,  # stop frequency (Hz)
            0,  # increment frequency per `dwell_time`
            0,  # dwell_time 
            ctypes.c_int32(0),  # sweep type
            0  # number of sweeps
        )
        assert_pico2000_ok(status)

    def _set_block_time(self, timebase, max_samples):
        from picosdk.ps2000 import ps2000
        from picosdk.functions import assert_pico2000_ok

        maxSamplesReturn = ctypes.c_int32()
        oversample = ctypes.c_int16(self._config.oversample)
        status = ps.ps2000_get_timebase(
            self.chandle,  # handle
            self._config.timebase,  # 0 is fastest, 2x slower each increment
            self._config.max_readings,  # number of readings
            ctypes.byref(self.timeInterval),  # pointer to time interval between readings (ns)
            ctypes.byref(self.timeUnits),  # pointer to time units
            oversample,  # on board averaging of concecutive `oversample` timepoints (increase resolution)
            ctypes.byref(maxSamplesReturn) # pointer to actual number of available samples
        )
        # todo: readout params on failure
        assert_pico2000_ok(status)

    async def _measure(self):
        return {"channel": 0}
