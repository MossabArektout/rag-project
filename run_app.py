"""
Launcher script for Smart Internship Assistant
Starts both API server and Streamlit interface
"""
import subprocess
import sys
import time
import webbrowser
from loguru import logger

def main():
    logger.info("Starting Smart Internship Assistant...")
    
    # Start API server
    logger.info("Starting API server on http://localhost:8000")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for API to start
    logger.info("Waiting for API to start...")
    time.sleep(3)
    
    # Start Streamlit
    logger.info("Starting Streamlit interface...")
    logger.info("Streamlit will open in your browser at http://localhost:8501")
    
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
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