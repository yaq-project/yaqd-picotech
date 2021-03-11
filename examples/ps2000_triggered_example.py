import ctypes
import numpy as np
from picosdk.ps2000 import ps2000 as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, mV2adc, assert_pico2000_ok

# working example

chARange = chBRange = 6
status = {}
wave_type = {
    k: i for i, k in enumerate(["sine", "square", "triangle", "ramp_up", "ramp_down", "dc"])
}

status["openUnit"] = ps.ps2000_open_unit()
assert_pico2000_ok(status["openUnit"])
chandle = ctypes.c_int16(status["openUnit"])
try:
    status["setsig"] = ps.ps2000_set_sig_gen_built_in(
        chandle, 500000, ctypes.c_uint32(1000000), 1, 1000, 1000, 0, 0, ctypes.c_int32(0), 0
    )
    assert_pico2000_ok(status["setsig"])
    status["setChA"] = ps.ps2000_set_channel(chandle, 0, 1, 1, chARange)
    assert_pico2000_ok(status["setChA"])
    status["setChB"] = ps.ps2000_set_channel(chandle, 1, 1, 1, chBRange)
    assert_pico2000_ok(status["setChB"])
    # find maximum ADC count value
    maxADC = ctypes.c_uint16(2 ** 15)  # ddk: counts appears ready for 16 bit signed...
    adc_threshold = mV2adc(500, chARange, maxADC)
    # Set number of pre and post trigger samples to be collected
    maxSamples = 200  # maxSamples * oversample < maxSamplesReturn = memory / num_channels

    status["set_trigger"] = ps.ps2000_set_trigger(chandle, 0, adc_threshold, 0, -50, 1)
    assert_pico2000_ok(status["set_trigger"])

    timebase = 5
    timeInterval = ctypes.c_int32()
    timeUnits = ctypes.c_int32()
    oversample = ctypes.c_int16(16)
    maxSamplesReturn = ctypes.c_int32()
    status["getTimebase"] = ps.ps2000_get_timebase(
        chandle,  # handle
        timebase,  # 0 is fastest, 2x slower each increment
        maxSamples,  # number of samples
        ctypes.byref(timeInterval),  # pointer to time interval
        ctypes.byref(timeUnits),  # pointer to time units
        oversample,
        ctypes.byref(maxSamplesReturn),
    )
    assert_pico2000_ok(status["getTimebase"])
    print(timeInterval, timeUnits, maxSamplesReturn)

    timeIndisposedms = ctypes.c_int32()
    status["runBlock"] = ps.ps2000_run_block(
        chandle, maxSamples, timebase, oversample, ctypes.byref(timeIndisposedms)
    )
    assert_pico2000_ok(status["runBlock"])

    # Check for data collection to finish using ps5000aIsReady
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps2000_ready(chandle)
        ready = ctypes.c_int16(status["isReady"])

    # Create buffers ready for data
    # bufferA = (ctypes.c_int16 * maxSamples)()
    # bufferB = (ctypes.c_int16 * maxSamples)()
    buffers = [None] * 4
    for i in range(len(["a", "b"])):
        buffers[i] = (ctypes.c_int16 * maxSamples)()

    # Get data from scope
    cmaxSamples = ctypes.c_int32(maxSamples)
    status["getValues"] = ps.ps2000_get_values(
        chandle,  # handle
        *[ctypes.byref(b) if b is not None else None for b in buffers],
        # ctypes.byref(bufferA),  # pointer to buffer_a
        # ctypes.byref(bufferB),  # pointer to buffer_b
        # None,  # pointer to buffer_c (NA)
        # None,  # pointer to buffer_d (NA)
        ctypes.byref(oversample),  # pointer to overflow
        cmaxSamples,  # number of values
    )
    assert_pico2000_ok(status["getValues"])

    bufferA = buffers[0]
    bufferB = buffers[1]
    # convert ADC counts data to mV
    adc2mVChA = adc2mV(bufferA, chARange, maxADC)
    adc2mVChB = adc2mV(bufferB, chBRange, maxADC)

    # Create time data
    time = np.linspace(0, (cmaxSamples.value) * timeInterval.value, cmaxSamples.value) / 1e6
    time -= time.mean()

    # plot data from channel A and B
    plt.plot(time, adc2mVChA[:])
    plt.plot(time, adc2mVChB[:])
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.show()

finally:
    status["stop"] = ps.ps2000_stop(chandle)
    assert_pico2000_ok(status["stop"])
    status["close"] = ps.ps2000_close_unit(chandle)
    assert_pico2000_ok(status["close"])
