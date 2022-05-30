from pkg_resources import DistributionNotFound, get_distribution

__all__ = []

try:
    __version__ = get_distribution(__name__).version
    __all__.append("__version__")
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass


del DistributionNotFound
del get_distribution
