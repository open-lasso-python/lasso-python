from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("lasso-python")
except PackageNotFoundError:
    # package is not installed
    pass
