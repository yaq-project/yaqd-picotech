__all__ = ["PicotechAdcTriggered"]

import asyncio
import ctypes
import numpy as np  # type: ignore
from dataclasses import dataclass
from time import sleep
import toml

from picosdk.functions import adc2mV, mV2adc  # type: ignore
from typing import Dict, Any, List
from yaqd_core import IsSensor, IsDaemon, HasMeasureTrigger

def process_samples(method, samples, axis=0):
    # samples arry shape: (sample, shot)
    if method == "average":
        shots = np.mean(samples, axis=axis)
    elif method == "sum":
        shots = np.sum(samples, axis=axis)
    elif method == "min":
        shots = np.min(samples, axis=axis)
    elif method == "max":
        shots = np.max(samples, axis=axis)
    else:
        raise KeyError("sample processing method not recognized")
    return shots


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
    "20 V": 10,
}

wave_type_to_code = {
    k: i for i, k in enumerate(["sine", "square", "triangle", "ramp_up", "ramp_down", "dc"])
}

# maximum ADC count value
# drivers normalize to 16 bit (15 bit signed) regardless of resolution
maxADC = ctypes.c_uint16(2 ** 15)
# ddk: ignore chopper class for now; only two physical channels to test


@dataclass
class Channel:
    name: str
    # label: str
    physical_channel: int
    signal_start: int
    signal_stop: int
    processing_method: str
    baseline_start: int
    baseline_stop: int
    range: str
    enabled: bool
    coupling: str
    invert: bool
    use_baseline: bool

    def volts_to_adc(self, x):
        return mV2adc(x * 1e3, range_to_code[self.range], maxADC)

    def adc_to_volts(self, x):
        return np.array(adc2mV(x, range_to_code[self.range], maxADC)) / 1e3


class PicotechAdcTriggered(HasMeasureTrigger, IsSensor, IsDaemon):
    # ddk:  order matters for base classes?
    _kind = "picotech-adc-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        print(toml.dumps(self._config))
        # print(self._config.items())

        self._channels = []
        for name, d in self._config["channels"].items():
            channel = Channel(**d, physical_channel="ABCD".index(name), name=name)
            # channel.label = channel.name if channel.label is None else name
            self._channels.append(channel)
        self._channel_names = [c.name for c in self._channels]  # expected by parent
        self._channel_units = {k: "V" for k in self._channel_names}  # expected by parent

        # check that all physical channels are unique
        x = []
        x += [c.physical_channel for c in self._channels]
        assert len(set(x)) == len(x)

        # only support ps2000 currently
        assert self._config["model"].lower() == "ps2000"

        # finish
        self._open_unit()
        self.state_change = False
        # self.measure_tries = 0
        self.measure(loop=self._config["loop_at_startup"])

    def _open_unit(self):
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        status = ps2000.ps2000_open_unit()
        assert_pico2000_ok(status)
        self.chandle = ctypes.c_int16(status)

        self._set_channels()
        self._set_block_time()
        if self._config["trigger_self"]:
            self._set_awg()
        self._set_trigger()

    def _set_channels(self):
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        for c in self._channels:
            status = ps2000.ps2000_set_channel(
                self.chandle,
                c.physical_channel,  # channel
                c.enabled,  # enabled
                c.coupling == "DC",  # dc (True) / ac (False)
                range_to_code[c.range],
            )
            assert_pico2000_ok(status)

    def _set_trigger(self):
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        trigger_channel = self._channels["ABCD".index(self._config["trigger_channel"])]
        status = ps2000.ps2000_set_trigger(
            self.chandle,
            trigger_channel.physical_channel,  # todo: convert to physical channel
            trigger_channel.volts_to_adc(self._config["trigger_threshold"] * 1e-6),  # threshold
            int(not self._config["trigger_rising"]),  # direction (0=rising, 1=falling)
            self._config["trigger_delay"],
            0,  # ms to wait before collecting if no trigger recieved (0 = infinity)
        )
        assert_pico2000_ok(status)

    def _set_awg(self):
        """
        generate 1 V p-p square wave at 1 kHz
        """
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        # awg
        status = ps2000.ps2000_set_sig_gen_built_in(
            self.chandle,
            500000,  # offset voltage (uV)
            ctypes.c_uint32(1000000),  # peak-to-peak voltage (uV)
            wave_type_to_code["square"],  # wavetype code
            1000,  # start frequency (Hz)
            1000,  # stop frequency (Hz)
            0,  # increment frequency per `dwell_time`
            0,  # dwell_time
            ctypes.c_int32(0),  # sweep type
            0,  # number of sweeps
        )
        assert_pico2000_ok(status)

    def _set_block_time(self):
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        maxSamplesReturn = ctypes.c_int32()
        oversample = ctypes.c_int16(self._config["oversample"])
        time_interval = ctypes.c_int32()
        time_units = ctypes.c_int32()

        status = ps2000.ps2000_get_timebase(
            self.chandle,  # handle
            self._config["timebase"],  # 0 is fastest, 2x slower each integer increment
            self._config["max_samples"],  # number of readings
            ctypes.byref(time_interval),  # pointer to time interval between readings (ns)
            ctypes.byref(time_units),  # pointer to time units
            oversample,  # on board averaging of concecutive `oversample` timepoints (increase resolution)
            ctypes.byref(maxSamplesReturn),  # pointer to actual number of available samples
        )
        self.time_interval = time_interval.value
        self.time_units = time_units.value
        """
        # ddk: time according to picotech example, but I believe spacing is off by 1/nsamples...
        self.time = np.linspace(
            0,
            self._config.max_samples * self.time_interval,
            self._config.max_samples
        ) / 1e6
        """
        self.time = np.arange(self._config["max_samples"], dtype=float) * self.time_interval
        # offset for delay
        self.time += self.time.max() * self._config["trigger_delay"] / 100
        # todo: readout params on failure
        assert_pico2000_ok(status)

    def _create_task(self):
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        time_indisposed_ms = ctypes.c_int32()
        # pointer to approximate time DAQ takes to collect data
        # i.e. (sample interval) x (number of points required)
        status = ps2000.ps2000_run_block(
            self.chandle,
            self._config["max_samples"],
            self._config["timebase"],
            ctypes.c_int16(self._config["oversample"]),
            ctypes.byref(time_indisposed_ms),
        )
        assert_pico2000_ok(status)
        self.time_indisposed = time_indisposed_ms.value / 1e3

    async def _measure(self):
        # self.measure_tries += 1
        # print("calling _measure", self.measure_tries)
        samples = await self._loop.run_in_executor(None, self._measure_samples)
        # shape: (nshots, samples)
        # print(samples["A"].shape)
        shots = {}
        # channels
        for channel in self._channels:
            if not channel.enabled:
                continue
            shots[channel.name] = np.empty((self._state["nshots"]))  # necessary?
            # signal:  collapse intra-shot (samples) time dimension
            signal_samples = samples[channel.name][
                :, channel.signal_start : channel.signal_stop + 1
            ]
            signal_shots = process_samples(channel.processing_method, signal_samples, axis=1)
            # baseline
            if not channel.use_baseline:
                shots[channel.name] = signal_shots
            else:
                baseline_samples = samples[channel.name][
                    :, channel.baseline_start : channel.baseline_stop + 1
                ]
                baseline_shots = process_samples(
                    channel.processing_method, baseline_samples, axis=1
                )
                shots[channel.name] = signal_shots - baseline_shots
            if channel.invert:
                shots[channel.name] *= -1
        # finish
        if self.state_change:
            self.state_change = False
            return self._measure()
        self._samples = samples
        self._shots = shots
        # collapse shots for readout:
        out = {c.name: process_samples(c.processing_method, shots[c.name]) for c in self._channels}
        return out

    def _measure_samples(self):
        """loop through shots, return aggregate"""
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        samples = {
            name: np.zeros((self._state["nshots"], self._config["max_samples"]), dtype=np.float)
            for name in self._channel_names
        }
        self._create_task()
        i = 0
        not_ready = 0
        while i < self._state["nshots"]:
            for wait in np.geomspace(self.time_indisposed, 60, num=15):
                status = ps2000.ps2000_ready(self.chandle)
                if status != not_ready:
                    assert_pico2000_ok(status)
                    break
                sleep(wait)
            else:
                # timeout; kill acquisition
                # todo: call ps2000_stop, re-initialize
                # from picosdk.ps2000 import ps2000  # type: ignore
                # from picosdk.functions import assert_pico2000_ok  # type: ignore

                # status = ps2000.ps2000_stop()
                # assert_pico2000_ok(status)
                return self._measure_samples()  # non-ideal: restarts all nshots if one fails
            sample = self._measure_sample()
            for name in self._channel_names:
                samples[name][i] = sample[name]
                # print(sample[name].min(), sample[name].max())
            if self.state_change:
                self.state_change = False
                return self._measure_samples()
            self._create_task()
            i += 1
        return samples

    def _measure_sample(self):
        """
        retrieve samples from single shot
        """
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        # create buffers for data
        buffers = [None] * 4
        for c in self._channels:
            if c.enabled:
                buffers[c.physical_channel] = (ctypes.c_int16 * self._config["max_samples"])()
        overflow = ctypes.c_int16()  # bit pattern on whether overflow has occurred

        status = ps2000.ps2000_get_values(
            self.chandle,  # handle
            # pointers to channel buffers
            *[ctypes.byref(b) if b is not None else None for b in buffers],
            ctypes.byref(overflow),  # pointer to overflow
            ctypes.c_int32(self._config["max_samples"]),  # number of values
        )
        assert_pico2000_ok(status)

        # todo: match physical channel to buffer; currently assumes order preserved
        sample = {c.name: c.adc_to_volts(b) for c, b in zip(self._channels, buffers)}
        # samples shape:  nsamples, shots
        return sample

    def get_sample_time(self):
        return self.time

    def get_channel_units(self):
        return "V"

    def get_measured_samples(self):
        """shape [channels, shots, samples]"""
        out = np.stack([arr for arr in self._samples.values()])
        # print("get_measured_samples")
        # print(out.shape, out[:,0].min())
        return out

    def get_measured_shots(self):
        """shape (channel, shot)"""
        out = np.stack([arr for arr in self._shots.values()])
        # print("get_measured_shots")
        # print(out.shape, out.dtype, out.min(), out.max())
        return out

    def get_nshots(self):
        return self._state["nshots"]

    def set_nshots(self, nshots):
        """Set number of shots."""
        assert nshots > 0
        self.state_change = True
        self._state["nshots"] = nshots
