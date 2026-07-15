import subprocess
import time
import webview
import sys

# Start the TrafficPulse viewer
server = subprocess.Popen([sys.executable, "viewer/app.py"])

# Give it a few seconds to start
time.sleep(3)

# Open a native desktop window
webview.create_window(
    "TrafficPulse",
    "http://127.0.0.1:8000",
    width=1400,
    height=900,
)

webview.start()

server.terminate()