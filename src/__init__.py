"""
Whaling Data Pipeline - Venture Capital of the Sea.

A comprehensive data pipeline for assembling, analyzing, and linking
American whaling voyage data with census records for captain wealth analysis.
"""

from . import _public_api as _public_api

PUBLIC_API = _public_api.PUBLIC_API
__all__ = _public_api.__all__
__version__ = _public_api.__version__

globals().update(PUBLIC_API)
