#!/usr/bin/env python3

import subprocess
import os
import sys
import socket
from contextlib import closing


def run_streamlit():
    """Run the streamlit app with proper configuration"""
    print("Starting Telecom AI Agent Streamlit UI...")
    
    # Add current directory to path (helps with imports)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Decide on a free port ourselves (some Streamlit versions don't log the auto-picked port correctly)
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    def find_free_port(preferred: int | None = None, start: int = 8501, end: int = 8599) -> int:
        """Find an available TCP port on localhost.
        Tries preferred first (if given), else scans a small range.
        """
        def is_free(p: int) -> bool:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    return True
                except OSError:
                    return False

        if isinstance(preferred, int) and preferred > 0:
            if is_free(preferred):
                return preferred
        for p in range(start, end + 1):
            if is_free(p):
                return p
        # Fallback: ask OS for an ephemeral port and use that
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    preferred_env = os.environ.get("STREAMLIT_SERVER_PORT")
    try:
        preferred_port = int(preferred_env) if preferred_env else None
    except ValueError:
        preferred_port = None
    port = find_free_port(preferred=preferred_port)
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    
    # Get the path to app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(current_dir, "app.py")  # unified app
    
    # Check that the file exists
    if not os.path.exists(ui_path):
        print(f"Error: UI file not found at {ui_path}")
        return
    
    # Start Streamlit
    print(f"Running Streamlit UI from: {ui_path}")
    print(f"Selected port: {port}")
    print(f"Local URL: http://localhost:{port}")
    # Use module invocation to avoid PATH issues on Windows
    # Try to start Streamlit; if port busy, retry with new port up to 5 times
    attempts = 0
    last_err = None
    while attempts < 5:
        attempts += 1
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run", ui_path,
                "--server.headless", os.environ["STREAMLIT_SERVER_HEADLESS"],
                "--server.port", str(port),
            ]
            print("Executing:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            # If the error likely due to port in use, pick another and retry
            print(f"Port {port} unavailable or Streamlit failed (exit code {e.returncode}). Retrying with a new port...")
            port = find_free_port(start=port + 1, end=port + 50)
            os.environ["STREAMLIT_SERVER_PORT"] = str(port)
            print(f"Selected new port: {port}")
            print(f"Local URL: http://localhost:{port}")
    # Exhausted retries
    print("Streamlit failed to start after multiple attempts. Ensure no other instance is running and dependencies are installed.")
    if last_err:
        raise last_err

if __name__ == "__main__":
    run_streamlit()
