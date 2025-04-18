from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lasso-python")
except PackageNotFoundError:
    # package is not installed
    pass
