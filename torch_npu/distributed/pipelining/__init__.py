from ._pipelining_patch import pipeline

pipeline.__module__ = __name__

__all__ = ["pipeline"]
