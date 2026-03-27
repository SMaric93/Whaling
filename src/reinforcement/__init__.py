"""
Reinforcement Test Suite for Whaling Organizational Search Paper.

Implements empirical tests to strengthen claims about:
1. Captains determine destination choice ("the map")
2. Organizations shape within-ground search behavior ("the compass")
3. Organizational capability reflects routines, not only physical capital
4. High-capability organizations raise the floor for novices
5. Production technology is submodular under scarcity
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "reinforcement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
