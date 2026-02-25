"""Entry point for PyInstaller-bundled Streamlit application.

Script ini adalah pintu masuk saat aplikasi dijalankan sebagai .exe.
Flow: .exe → run_streamlit_app.py → streamlit.web.cli.main() → app.py
"""

import os
import sys
import threading
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def setup_paths():
    """Setup sys.path for frozen (PyInstaller) and normal execution."""
    if getattr(sys, 'frozen', False):
        # Running as bundled .exe
        if hasattr(sys, '_MEIPASS'):
            # --onefile mode
            base_dir = Path(sys._MEIPASS)
        else:
            # --onedir mode
            base_dir = Path(sys.executable).resolve().parent
        
        program_dir = base_dir / "Program"
        gui_dir = program_dir / "gui"
        tabs_dir = gui_dir / "tabs"

        # Add all necessary paths
        for p in [str(program_dir), str(gui_dir), str(tabs_dir), str(base_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Set working directory to base_dir so Streamlit can find its assets
        os.chdir(str(base_dir))

        return str(gui_dir / "app.py")
    else:
        # Running as normal script
        base_dir = Path(__file__).resolve().parent
        program_dir = base_dir / "Program"
        gui_dir = program_dir / "gui"
        tabs_dir = gui_dir / "tabs"

        for p in [str(program_dir), str(gui_dir), str(tabs_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        return str(gui_dir / "app.py")


def run_streamlit(app_script, port):
    """Run Streamlit in a separate thread."""
    import signal
    # Monkeypatch signal.signal to avoid "signal only works in main thread" error
    # since we are running in a background thread
    signal.signal = lambda *args, **kwargs: None
    
    try:
        from streamlit.web import cli as stcli
        sys.argv = [
            "streamlit", "run", app_script,
            "--server.address", "localhost",
            "--server.port", str(port),
            "--server.headless", "true",
            "--global.developmentMode", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
        ]
        stcli.main()
    except Exception as e:
        print(f"\n[ERROR] Streamlit server failed: {e}")


def main():
    """Launch the application with pywebview on main thread."""
    print("=" * 60)
    print("  MFVRPTW | Route Optimizer")
    print("  Memulai aplikasi...")
    print("=" * 60)

    app_script = setup_paths()
    
    if not Path(app_script).exists():
        print(f"\n[ERROR] File aplikasi tidak ditemukan: {app_script}")
        input("\nTekan Enter untuk keluar...")
        sys.exit(1)

    port = 8501
    url = f"http://localhost:{port}"

    # 1. Jalankan Streamlit di background thread
    st_thread = threading.Thread(
        target=run_streamlit,
        args=(app_script, port),
        daemon=True
    )
    st_thread.start()

    # 2. Tunggu sebentar agar server naik
    import time
    print("[...] Menunggu server siap...")
    time.sleep(5)

    # 3. Jalankan GUI di main thread (Wajib untuk Windows/macOS)
    try:
        import webview
        print(f"[>>] Membuka jendela desktop: MFVRPTW | Route Optimizer")
        webview.create_window(
            "MFVRPTW | Route Optimizer | Nabilah Eva Nurhayati", 
            url, 
            width=1280, 
            height=800, 
            min_size=(800, 600)
        )
        webview.start()
    except Exception as e:
        print(f"[!] Gagal membuka GUI: {e}")
        print(f"[!] Fallback: Silakan buka {url} di browser.")
        # Kita tidak panggil webbrowser.open otomatis agar user tahu ada yang salah dengan GUI-nya
        input("\nTekan Enter untuk keluar...")


if __name__ == "__main__":
    main()
