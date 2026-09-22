"""NumPy-only Gemma 4 12B runtime."""
import os as _os

# The attention uses small matrix products. The BLAS library must use one
# thread for them. Many BLAS threads fight the OpenMP threads of the int8
# kernel and make a long decode slow. Set the values before NumPy loads.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "6")

from .config import Config
from .st import SafeTensors
from .model import Model, KVCache, Session
from .tokenizer import Tokenizer

__all__ = ["Config", "SafeTensors", "Model", "KVCache", "Session", "Tokenizer"]
