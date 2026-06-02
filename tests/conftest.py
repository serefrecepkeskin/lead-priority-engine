"""Test bootstrap.

The data-generation code lives under ``scripts/`` (not in the runtime
package), so we prepend ``scripts/`` to ``sys.path`` here. CLI scripts
already see it automatically when run directly; tests and the notebook
need this nudge.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
