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
        [ndarray (channels), list of channel names, list of channel units]
    """
    out = [arr.mean() for arr in arrs]
    out_names = [name + "_mean" for name in names]
    out_units = [unit for unit in units]
    return [out, out_names, out_units]
