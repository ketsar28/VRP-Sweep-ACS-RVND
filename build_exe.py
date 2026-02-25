"""Build script for creating PyInstaller .exe of MFVRPTW Optimizer.

Usage: python build_exe.py
Output: dist/MFVRPTW_Route Optimizer_Nabilah Eva Nurhayati/MFVRPTW_Route Optimizer_Nabilah Eva Nurhayati.exe
"""

import os
import sys
import shutil
import subprocess

# Fix Windows console encoding for emoji
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
ENTRY_POINT = ROOT / "run_streamlit_app.py"
APP_NAME = "MFVRPTW_Route Optimizer_Nabilah Eva Nurhayati"


def kill_processes():
    """Kill any running instances of the app to release file locks."""
    print("🔪 Killing old processes...")
    try:
        # Kill the app itself (old and new names)
        # We use taskkill on Windows for aggressive termination
        process_names = [
            "MFVRPTW_Optimizer.exe",
            "MFVRPTW_Route Optimizer_Nabilah Eva Nurhayati.exe",
            "webview.exe",
            "python.exe" # Be careful with this, but needed for some zombie streamlit processes
        ]
        
        # Get list of running processes to be more surgical
        try:
            output = subprocess.check_output('tasklist /FI "STATUS eq RUNNING" /FO CSV', shell=True).decode()
            for name in process_names:
                if name.lower() in output.lower():
                    print(f"   - Terminating {name}...")
                    subprocess.run(f'taskkill /F /IM "{name}" /T', shell=True, capture_output=True)
        except Exception:
            # Fallback to simple taskkill calls
            for name in process_names:
                subprocess.run(f'taskkill /F /IM "{name}" /T', shell=True, capture_output=True)
        
        # Small sleep to let OS release file handles
        import time
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Warning during cleanup: {e}")


def clean():
    """Remove old build artifacts with retry logic."""
    kill_processes()
    for d in [DIST_DIR, BUILD_DIR]:
        if d.exists():
            print(f"🧹 Cleaning {d}...")
            # Try to remove multiple times if needed (Windows is slow at releasing handles)
            import time
            for attempt in range(3):
                try:
                    shutil.rmtree(d)
                    break
                except PermissionError as e:
                    if attempt < 2:
                        print(f"   ⏳ File locked, retrying in 1s... ({e.filename})")
                        time.sleep(1)
                    else:
                        print(f"❌ ERROR: Cannot delete {d}. Close any open folders or apps and try again.")
                        print(f"   Detail: {e}")
                        sys.exit(1)
                except Exception as e:
                    print(f"   ⚠️ Could not fully clean {d}: {e}")
                    break


def get_streamlit_path():
    """Find where streamlit is installed."""
    import streamlit
    return Path(streamlit.__file__).parent


def build():
    """Run PyInstaller with all necessary options."""
    print("=" * 60)
    print(f"  Building {APP_NAME}")
    print("=" * 60)

    streamlit_path = get_streamlit_path()
    print(f"📦 Streamlit found at: {streamlit_path}")

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",

        # Use --onedir (more reliable for Streamlit)
        "--onedir",

        # Keep console window open for debugging
        "--console",

        # ===== EXCLUDE unnecessary/conflicting packages =====
        # Qt bindings (Streamlit uses Tornado, not Qt)
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        "--exclude-module", "PySide2",
        "--exclude-module", "PySide6",
        "--exclude-module", "qtpy",
        # GUI toolkits
        "--exclude-module", "tkinter",
        "--exclude-module", "_tkinter",
        # Heavy ML/AI frameworks (not used by this app)
        "--exclude-module", "tensorflow",
        "--exclude-module", "keras",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "transformers",
        "--exclude-module", "xgboost",
        "--exclude-module", "lightgbm",
        "--exclude-module", "catboost",
        "--exclude-module", "sklearn",
        "--exclude-module", "scikit-learn",
        # Distributed computing
        "--exclude-module", "dask",
        "--exclude-module", "distributed",
        "--exclude-module", "numba",
        "--exclude-module", "llvmlite",
        # Jupyter/IPython ecosystem
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "--exclude-module", "jupyter_client",
        "--exclude-module", "jupyter_core",
        "--exclude-module", "notebook",
        "--exclude-module", "ipykernel",
        "--exclude-module", "ipywidgets",
        "--exclude-module", "nbformat",
        "--exclude-module", "nbconvert",
        # Documentation/testing
        "--exclude-module", "sphinx",
        "--exclude-module", "docutils",
        "--exclude-module", "test",
        "--exclude-module", "unittest",
        "--exclude-module", "pytest",
        # Other heavy unused packages
        "--exclude-module", "cv2",
        "--exclude-module", "PIL.ImageTk",
        "--exclude-module", "matplotlib.backends.backend_tkagg",
        "--exclude-module", "botocore",
        "--exclude-module", "boto3",
        "--exclude-module", "sqlalchemy",
        "--exclude-module", "tables",
        "--exclude-module", "h5py",
        "--exclude-module", "zmq",
        "--exclude-module", "lxml",
        "--exclude-module", "astroid",
        "--exclude-module", "pylint",

        # ===== DATA FILES =====
        # Bundle the entire Program/ folder
        "--add-data", f"Program{os.pathsep}Program",

        # Bundle Streamlit's static files (critical!)
        "--add-data", f"{streamlit_path / 'static'}{os.pathsep}streamlit/static",
        "--add-data", f"{streamlit_path / 'runtime'}{os.pathsep}streamlit/runtime",

        # ===== COLLECT ALL =====
        "--collect-all", "streamlit",
        "--collect-all", "plotly",
        "--collect-all", "altair",
        "--collect-all", "webview",
        "--collect-all", "pythonnet",

        # ===== HIDDEN IMPORTS =====
        # Streamlit core
        "--hidden-import", "streamlit",
        "--hidden-import", "streamlit.web",
        "--hidden-import", "streamlit.web.cli",
        "--hidden-import", "streamlit.web.server",
        "--hidden-import", "streamlit.web.server.server",
        "--hidden-import", "streamlit.runtime",
        "--hidden-import", "streamlit.runtime.scriptrunner",
        "--hidden-import", "streamlit.runtime.scriptrunner.script_runner",
        "--hidden-import", "streamlit.runtime.caching",
        "--hidden-import", "streamlit.runtime.state",
        "--hidden-import", "streamlit.runtime.uploaded_file_manager",
        "--hidden-import", "streamlit.runtime.media_file_manager",
        "--hidden-import", "streamlit.runtime.memory_uploaded_file_manager",
        "--hidden-import", "streamlit.commands.page_config",
        "--hidden-import", "streamlit.components.v1",
        "--hidden-import", "streamlit.elements",
        "--hidden-import", "streamlit.delta_generator",
        "--hidden-import", "streamlit.config",

        # Streamlit dependencies
        "--hidden-import", "validators",
        "--hidden-import", "validators.url",
        "--hidden-import", "toml",
        "--hidden-import", "watchdog",
        "--hidden-import", "watchdog.observers",
        "--hidden-import", "watchdog.events",
        "--hidden-import", "gitdb",
        "--hidden-import", "gitdb.db.loose",
        "--hidden-import", "gitdb.db.pack",
        "--hidden-import", "smmap",
        "--hidden-import", "tornado",
        "--hidden-import", "tornado.web",
        "--hidden-import", "tornado.websocket",
        "--hidden-import", "click",
        "--hidden-import", "rich",
        "--hidden-import", "rich.console",
        "--hidden-import", "pympler",
        "--hidden-import", "pydeck",
        "--hidden-import", "pyarrow",
        "--hidden-import", "cachetools",
        "--hidden-import", "tenacity",

        # Plotly
        "--hidden-import", "plotly",
        "--hidden-import", "plotly.graph_objects",
        "--hidden-import", "plotly.express",
        "--hidden-import", "plotly.subplots",
        "--hidden-import", "plotly.io",
        "--hidden-import", "plotly.io._renderers",

        # Data libraries
        "--hidden-import", "pandas",
        "--hidden-import", "pandas.core",
        "--hidden-import", "pandas.io",
        "--hidden-import", "pandas.io.formats",
        "--hidden-import", "pandas.io.formats.style",
        "--hidden-import", "numpy",
        "--hidden-import", "numpy.core",
        "--hidden-import", "scipy",
        "--hidden-import", "openpyxl",
        "--hidden-import", "xlsxwriter",

        # Other dependencies
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "packaging",
        "--hidden-import", "packaging.version",
        "--hidden-import", "packaging.requirements",
        "--hidden-import", "importlib_metadata",
        "--hidden-import", "charset_normalizer",
        "--hidden-import", "certifi",
        "--hidden-import", "urllib3",
        "--hidden-import", "requests",
        "--hidden-import", "typing_extensions",
        "--hidden-import", "tzdata",
        "--hidden-import", "pytz",
        "--hidden-import", "dateutil",
        "--hidden-import", "webview",
        "--hidden-import", "pythonnet",
        "--hidden-import", "clr_loader",

        # Protobuf (used by Streamlit)
        "--hidden-import", "google.protobuf",
        "--hidden-import", "google.protobuf.descriptor",
        "--hidden-import", "google.protobuf.internal",

        # Our application modules
        "--hidden-import", "path_helper",
        "--hidden-import", "sweep_nn",
        "--hidden-import", "acs_solver",
        "--hidden-import", "rvnd",
        "--hidden-import", "final_integration",
        "--hidden-import", "distance_time",
        "--hidden-import", "academic_replay",

        # Entry point
        str(ENTRY_POINT),
    ]

    print(f"\n🔨 Running PyInstaller...")
    print(f"   Command: {' '.join(cmd[:10])}...\n")

    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"\n❌ Build GAGAL! Exit code: {result.returncode}")
        sys.exit(1)

    # Verify output
    exe_path = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Build SUKSES!")
        print(f"   📁 Output: {exe_path}")
        print(f"   📏 Size: {size_mb:.1f} MB")
        print(f"\n   Jalankan dengan: {exe_path}")
    else:
        print(f"\n❌ File .exe tidak ditemukan di: {exe_path}")
        sys.exit(1)


if __name__ == "__main__":
    clean()
    build()
