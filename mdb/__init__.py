__version__ = "0.0.4"


def asmpca(*args, **kwargs):
    from mdb.asmpca import asmpca as _asmpca

    return _asmpca(*args, **kwargs)


def stats(*args, **kwargs):
    from mdb.stats import stats as _stats

    return _stats(*args, **kwargs)


def viz(*args, **kwargs):
    from mdb.viz import viz as _viz

    return _viz(*args, **kwargs)


def plot(*args, **kwargs):
    from mdb.plot import plot as _plot

    return _plot(*args, **kwargs)


__all__ = ["__version__", "asmpca", "stats", "viz", "plot"]
