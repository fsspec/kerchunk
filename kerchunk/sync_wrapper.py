import inspect
from fsspec.asyn import AsyncFileSystem


class SynchronousWrapper(AsyncFileSystem):
    """If the calling code requires an async filesystem, but you have sync"""

    def __init__(self, fs, **kw):
        self.fs = fs
        super().__init__(**kw)

    def __getattribute__(self, item):
        fs = object.__getattribute__(self, "fs")
        if inspect.iscoroutinefunction(getattr(AsyncFileSystem, item)):

            async def f(*args, **kwargs):
                return getattr(fs, item.lstrip("_"))(*args, **kwargs)

            return f
        else:
            return getattr(fs, item)
