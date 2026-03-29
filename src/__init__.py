"""
Whaling Data Pipeline - Venture Capital of the Sea.

A comprehensive data pipeline for assembling, analyzing, and linking
American whaling voyage data with census records for captain wealth analysis.
"""

from ._public_api import PUBLIC_API, __all__, __version__

globals().update(PUBLIC_API)
