import pathlib
import numpy as np


def process(arrs, names, units):
    """
    For each channel overall mean

    Parameters
    ----------
    arrs : list
        list of ndarrays (shots, time) for each raw channel
    names : list of str
        A list of input names for each raw channel
    units : list of units
        unit of each arrs for each raw channel

    Returns
    -------
    list
        [ndarray (channels), list of channel names, list of channel units, list of mappings (optional)]
    """
    # isolate b, a shots (proof of principle)
    # for now, assume chB is chopper, chA is signal
    chopper = names.index("chB")
    a = arrs[chopper] > 1.2 # a is on, b is off
    # b = ~a
    A_diff = arrs[0][chopper].mean() - arrs[0][~chopper].mean()
    
    out = [arr.mean() for arr in arrs]
    out += A_diff
    out_names = [name + "_mean" for name in names]
    out_names += names[0] + "_diff"
    out_units = {name: "V" for name in out_names}
    return [out, out_names, out_units]


