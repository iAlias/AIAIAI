"""Configurazione pytest: rende importabile il pacchetto 'italian_llm' dalla src layout."""

import os
import sys

# Radice del repo = cartella che contiene questa directory 'tests'.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")

# Inserisce src/ in testa al path cosi' 'import italian_llm' funziona senza install.
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
