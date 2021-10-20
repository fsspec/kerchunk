# old package alias, to be deprecated
import importlib

from kerchunk import *


def __getattr__(name):
    # aliases
    return importlib.import_module(f"kerchunk.{name}")

