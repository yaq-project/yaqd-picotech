import numpy as np


def process(arrs: dict, names, units):
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
    # for now, assume chB is chopper, chA is signal
    on = arrs["B"] > 1.2  # a is on, b is off
    # b = ~a
    A_diff = arrs["A"][on].mean() - arrs["A"][~on].mean()

    # counting pulses
    d_aa = np.sign(np.diff(arrs["A"], axis=1, prepend=0))
    dd_aa = np.diff(d_aa, append=0)
    count_rising = ((dd_aa < 0) & (d_aa == 1)).astype(np.int8)
    # count_falling = ((dd_aa > 0) & (d_aa==-1)).astype(np.int8)

    A_count = count_rising.sum()
    count_rising[~on] *= -1
    A_count_diff = count_rising.sum()

    out = {name + "_mean": arr.mean() for name, arr in zip(names, arrs)}
    out[names[0] + "_diff"] = A_diff
    out[names[0] + "_count"] = A_count
    out[names[0] + "_count_diff"] = A_count_diff

    out_units = {name: None if "count" in name else "V" for name in out.keys()}
    return [list(out.values()), list(out.keys()), out_units]
