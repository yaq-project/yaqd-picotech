__all__ = ["PicotechAdcTriggered"]

import asyncio
import ctypes
import numpy as np  # type: ignore
from dataclasses import dataclass
from time import sleep, time
import pathlib
import importlib.util
import sys

from picosdk.functions import adc2mV, mV2adc  # type: ignore
from typing import Dict, Any, List
from yaqd_core import IsSensor, IsDaemon, HasMeasureTrigger, HasMapping


# todo: parse range codes based on psx000.PSx000_VOLTAGE_RANGE dict
# ranges = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
# code_to_range = {i+1: ranges[i] for i in range(len(ranges))}
range_to_code = {
    "_20_mV": 1,
    "_50_mV": 2,
    "_100_mV": 3,
    "_200_mV": 4,
    "_500_mV": 5,
    "_1_V": 6,
    "_2_V": 7,
    "_5_V": 8,
    "_10_V": 9,
    "_20_V": 10,
}

wave_type_to_code = {
    k: i for i, k in enumerate(["sine", "square", "triangle", "ramp_up", "ramp_down", "dc"])
}

# maximum ADC count value
# drivers normalize to 16 bit (15 bit signed) regardless of resolution
maxADC = ctypes.c_uint16(2**15)


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class RawChannel:
    name: str
    index: int
    range: str
    enabled: bool
    coupling: str
    invert: bool

    def volts_to_adc(self, x):
        return mV2adc(x * 1e3, range_to_code[self.range], maxADC)

    def adc_to_volts(self, x):
        return np.array(adc2mV(x, range_to_code[self.range], maxADC)) / 1e3


class PicotechAdcTriggered(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    _kind = "picotech-adc-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._raw_channels = []
        self._raw_enabled_channels = []
        for name, d in self._config["channels"].items():
            channel = RawChannel(**d, index="ABCD".index(name), name=name)
            self._raw_channels.append(channel)
            if channel.enabled:
                self._raw_enabled_channels.append(channel)
        self._raw_channel_names = [c.name for c in self._raw_enabled_channels]
        self._raw_channel_units = {k: "V" for k in self._channel_names}
        self._raw_inverts = [[1, -1][c.invert] for c in self._raw_enabled_channels]
        self._samples = {}

        # ddk: I believe these are the native units of all models
        self._mapping_units["time"] = "ns"

        # check that all physical channels are unique
        x = []
        x += [c.index for c in self._raw_channels]
        assert len(set(x)) == len(x)

        # only support ps2xxx currently
        assert self._config["model"].lower().startswith("ps2")

        # processing module
        path = pathlib.Path(self._config["shots_processing_path"])
        if not path.is_absolute():
            path = (pathlib.Path(config_filepath).parent / path).resolve()
        self.processing_module = import_from_path("processing_module", path)

        # finish
        self._open_unit()
        self.state_change = False
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

        # enable channel if it is trigger?
        for c in self._raw_channels:
            if c.enabled or (
                self._config["trigger_self"] and self._config["trigger_channel"] == c.name
            ):
                enabled = True
            else:
                enabled = False
            self.logger.debug(f"{c.name}, {c.enabled}, {enabled}")
            status = ps2000.ps2000_set_channel(
                self.chandle,
                c.index,  # channel
                enabled,
                c.coupling == "DC",  # dc (True) / ac (False)
                range_to_code[c.range],
            )
            assert_pico2000_ok(status)

    def _set_trigger(self):
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        trigger_channel = self._raw_channels["ABCD".index(self._config["trigger_channel"])]
        status = ps2000.ps2000_set_trigger(
            self.chandle,
            trigger_channel.index,  # todo: convert to physical channel
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
        """
        # ddk: time according to picotech example, but I believe spacing is off by 1/nsamples...
        self.time = np.linspace(
            0,
            self._config.max_samples * self.time_interval,
            self._config.max_samples
        ) / 1e6
        """
        time = np.arange(self._config["max_samples"], dtype=float) * self.time_interval
        # offset for delay
        time += time.max() * self._config["trigger_delay"] / 100
        self._mappings["time"] = time

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
        self.time_indisposed = max(time_indisposed_ms.value / 1e3, 1e-5)

    async def _measure(self):
        start = time()
        samples = await self._loop.run_in_executor(None, self._measure_samples)
        finish = time()
        self.logger.debug(f"samples acquired {(finish - start):0.4f}")
        # samples value shapes: (nshots, samples)
        # invert
        for k, inv in zip(samples.keys(), self._raw_inverts):
            self._samples[k] = samples[k] * inv
        # filter samples
        if self._state["threshold_enabled"]:
            ignore = self._samples["A"] < self._state["threshold"]
            self.logger.info(f"{ignore.sum()=}")
            self._samples["A"][ignore] = 0
        # process
        self.logger.info(f"{self.raw_channel_names=}")
        try:
            out = self.processing_module.process(
                self._samples, self._raw_channel_names, self._raw_channel_units
            )
        except Exception as e:
            self.logger.error(e, stack_info=True)
            raise e
        if len(out) == 4:
            out_sig, out_names, out_units, out_mappings = out
        else:
            out_sig, out_names, out_units = out
            out_mappings = {name: [] for name in out_names}
        if self._measurement_id == 0:
            self._channel_names = out_names
            self._channel_units = out_units
            self._channel_mappings = out_mappings
            for k, v in zip(out_names, out_sig):
                self._channel_shapes[k] = [] if type(v) in [float, int] else v.shape
        # finish
        if self.state_change:
            self.state_change = False
            return self._measure()
        return {k: v for k, v in zip(out_names, out_sig)}

    def _measure_samples(self):
        # TODO: make asynchronous
        """
        loop through shots

        returns
        -------
            dict key:channel, value: ndarray[shot][sample]
        """
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        samples = {
            c.name: np.zeros((self._state["nshots"], self._config["max_samples"]), dtype=float)
            for c in self._raw_enabled_channels
        }
        self._create_task()

        # TODO: loop here is blocking; I should be feeding a queue
        for i in range(self._state["nshots"]):
            while True:
                status = ps2000.ps2000_ready(self.chandle)
                if status != 0:  # not_ready = 0
                    assert_pico2000_ok(status)
                    break
                if self.time_indisposed >= 0.1:
                    sleep(self.time_indisposed)
            sample = self._measure_sample()
            for name in self._raw_channel_names:
                samples[name][i] = sample[name]
            if self.state_change:
                self.state_change = False
                return self._measure_samples()
            self._create_task()
        return samples

    def _measure_sample(self) -> Dict[str, List]:
        """
        retrieve samples from single shot
        """
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok  # type: ignore

        buffers = [(ctypes.c_int16 * self._config["max_samples"])() for _ in range(4)]
        overflow = ctypes.c_int16()  # bit pattern on whether overflow has occurred

        status = ps2000.ps2000_get_values(
            self.chandle,  # handle
            *[ctypes.byref(b) for b in buffers],
            ctypes.byref(overflow),  # pointer to overflow
            ctypes.c_int32(self._config["max_samples"]),  # number of values
        )
        assert_pico2000_ok(status)
        sample = {c.name: c.adc_to_volts(buffers[c.index]) for c in self._raw_enabled_channels}
        # samples shape:  nsamples, shots
        return sample

    def close(self) -> None:
        self.stop_looping()
        from picosdk.ps2000 import ps2000  # type: ignore
        from picosdk.functions import assert_pico2000_ok, PicoSDKCtypesError  # type: ignore

        while True:
            try:
                status = ps2000.ps2000_close_unit(self.chandle)
                assert_pico2000_ok(status)
            except PicoSDKCtypesError:
                print("close failed with status; retrying")
                sleep(0.1)
            else:
                break
        return

    def get_measured_samples(self) -> np.ndarray:
        """shape [channels, shots, samples]"""
        out = np.stack([arr for arr in self._samples.values()])
        return out

    def get_nshots(self) -> int:
        return self._state["nshots"]

    def set_nshots(self, nshots) -> None:
        """Set number of shots."""
        assert nshots > 0
        self.state_change = True
        self._state["nshots"] = nshots

    def enable_threshold(self, flag: bool):
        self._state["threshold_enabled"] = flag

    def get_threshold(self) -> float:
        return self._state["threshold"]

    def set_threshold(self, val: float):
        self._state["threshold"] = val


if __name__ == "__main__":
    PicotechAdcTriggered.main()
