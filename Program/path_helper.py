"""Centralized path resolution for PyInstaller compatibility.

Semua module di folder Program/ harus import dari sini untuk mendapatkan
DATA_DIR dan PROGRAM_DIR yang benar, baik saat jalan sebagai script maupun
saat di-bundle sebagai .exe oleh PyInstaller.
"""

import sys
from pathlib import Path


def get_program_dir() -> Path:
    """Return path ke folder Program/.

    - Frozen (onedir): sys.executable ada di dist/AppName/ → Program/ ada di sebelahnya
    - Frozen (onefile): sys._MEIPASS berisi extracted files → Program/ ada di dalamnya
    - Script mode: Path(__file__) ada di Program/ → langsung parent
    """
    if getattr(sys, 'frozen', False):
        # Running as bundled .exe
        if hasattr(sys, '_MEIPASS'):
            # --onefile mode: extracted to temp dir
            return Path(sys._MEIPASS) / "Program"
        else:
            # --onedir mode: .exe sits next to Program/ folder
            return Path(sys.executable).resolve().parent / "Program"
    else:
        # Running as normal Python script
        return Path(__file__).resolve().parent


def get_data_dir() -> Path:
    """Return path ke folder Program/data/processed/."""
    return get_program_dir() / "data" / "processed"


def get_docs_dir() -> Path:
    """Return path ke folder Program/docs/."""
    return get_program_dir() / "docs"


def get_samples_dir() -> Path:
    """Return path ke folder Program/data/samples/."""
    return get_program_dir() / "data" / "samples"


# Convenience constants (evaluated at import time)
PROGRAM_DIR = get_program_dir()
DATA_DIR = get_data_dir()
DOCS_DIR = get_docs_dir()
SAMPLES_DIR = get_samples_dir()
