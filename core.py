"""Backward-compatible module: the original ``from core import ...`` keeps working.

The implementation now lives in ``experimentlab/core.py``.
"""
from experimentlab.core import *  # noqa: F401,F403
from experimentlab.core import REQUIRED, analyze, generate, sample_size, validate, wilson  # noqa: F401
