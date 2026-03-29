"""Compatibility package for environments that put `<repo>/src` on ``sys.path``.

Some modules/tests prepend the source directory itself to ``sys.path`` and then
import ``src``. In that layout, Python resolves ``src`` as ``<repo>/src/src``.
This shim re-exports the canonical public API and extends ``__path__`` so
``src.<submodule>`` continues to resolve to modules under ``<repo>/src``.
"""

from pathlib import Path

# Allow submodule imports like `src.analyses` to resolve from `<repo>/src`.
_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in __path__:
    __path__.append(str(_PARENT))

from src._public_api import PUBLIC_API, __all__, __version__  # type: ignore

globals().update(PUBLIC_API)
