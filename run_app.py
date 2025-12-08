"""
Launcher script for Smart Internship Assistant
Starts both API server and Streamlit interface
"""
import subprocess
import sys
import time
import webbrowser
import requests
from loguru import logger

def wait_for_api(url="http://localhost:8000/api/health", timeout=30, retry_interval=1):
    """Wait for API to become available"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.success("✓ API server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(retry_interval)

    logger.error("✗ API server failed to start within timeout period")
    return False

def main():
    logger.info("Starting Smart Internship Assistant...")

    # Start API server
    logger.info("Starting API server on http://localhost:8000")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )

    # Wait for API to be ready
    logger.info("Waiting for API to start...")
    if not wait_for_api():
        logger.error("Failed to start API server. Exiting...")
        api_process.terminate()
        return
    
    # Start Streamlit
    logger.info("Starting Streamlit interface...")
    logger.info("Streamlit will open in your browser at http://localhost:8501")
    
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    
    # Wait a bit then open browser
    time.sleep(2)
    webbrowser.open("http://localhost:8501")
    
    try:
        logger.info("Applications running. Press Ctrl+C to stop both.")
        # Wait for processes
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_process.terminate()
        streamlit_process.terminate()
        logger.success("Applications stopped.")

if __name__ == "__main__":
    main()