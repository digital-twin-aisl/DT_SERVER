"""Make shared source packages importable from a repository checkout.

Installed deployments receive ``dt-common`` as a normal Python package.  A
source checkout also needs to expose its ``src`` directory so application
entrypoints keep working when invoked directly from their own directories.
"""

from pathlib import Path
import sys


_COMMON_SRC = (
    Path(__file__).resolve().parents[1] / "packages" / "dt_common" / "src"
)
if _COMMON_SRC.is_dir() and str(_COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(_COMMON_SRC))
