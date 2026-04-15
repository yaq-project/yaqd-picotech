import pathlib


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
    A_diff = arrs[0][on].mean() - arrs[0][~on].mean()

    out = [arr.mean() for arr in arrs]
    out += A_diff
    out_names = [name + "_mean" for name in names]
    out_names += names[0] + "_diff"
    out_units = {name: "V" for name in out_names}
    return [out, out_names, out_units]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    here = pathlib.Path(__file__).resolve().parent
    saved = here / "data.npz"
    if saved.exists():
        arrs = np.load(saved)
        t = arrs["t"]
        chA = arrs["chA"]
        chB = arrs["chB"]
    else:
        data_paths = list((here / "20260324").glob("*.csv"))

        def convert_infinity(s):
            # Decode if necessary (if encoding='bytes' is used)
            s = s.decode("utf-8") if isinstance(s, bytes) else s
            s = s.strip()
            if s == "∞" or s.lower() == "inf":
                return float("inf")
            elif s == "-∞" or s.lower() == "-inf":
                return float("-inf")
            return float(s)

        datas = []
        for data_path in (here / "20260324").glob("*.csv"):
            print(data_path.name)
            datas.append(
                np.loadtxt(
                    data_path,
                    unpack=True,
                    dtype=float,
                    skiprows=3,
                    delimiter=",",
                    converters=convert_infinity,
                )
            )
        datas = np.array(datas)
        print(datas.shape)
        t, chA, chB = np.unstack(datas, axis=1)
        np.savez(saved, t=t, chA=chA, chB=chB)

    chB -= 2.5
    threshold = 20

    # detect counts to reduce noise thresholds
    if False:
        chA[-chA < threshold] = 0
        mean = chA.mean(axis=0)
        mean = np.convolve(mean, np.ones(10) / 10, mode="same")
        fig, ax = plt.subplots()
        # ax.plot(t, chA, c="k", alpha=0.2) # .mean(axis=0))
        ax.plot(t[0], mean, c="r", lw=1)  # .mean(axis=0))
        ax.plot(t, chB, c="b", alpha=0.1)  # .mean(axis=0))
        # ax.set_ylim(-30,30)
        fig.savefig(here / "test.png")
    if True:  # detect photon events, try to treat them all as equal
        # chA[-chA < threshold] = 0
        chA[chA == -np.inf] = -250
        grad = np.gradient(chA, axis=1)
        grad[grad > -10] = 0
        grad[grad <= -10] = 1

        mean = grad.mean(axis=0)
        mean = np.convolve(mean, np.ones(10) / 10, mode="same")

        fig, ax = plt.subplots()
        # ax.plot(t[0], mean, c="k")
        # ax.plot(t[0], grad[2], c="k")
        ax.plot(t[0], chA[3], c="k")
        # ax.set_xlim(-5,5)

        fig.savefig(here / "test_edge_detect.png", dpi=200)
