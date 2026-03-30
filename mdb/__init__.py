__version__ = "0.0.3"


def asmpca(*args, **kwargs):
    from mdb.asmpca import asmpca as _asmpca

    return _asmpca(*args, **kwargs)


__all__ = ["__version__", "asmpca"]
