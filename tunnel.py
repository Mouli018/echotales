"""
EchoTales — ngrok tunnel launcher
Run this to expose your local Streamlit app to the internet:

    python tunnel.py

You will need a free ngrok account and authtoken.
Get one at: https://dashboard.ngrok.com/get-started/your-authtoken
"""

import subprocess
import sys
import time
import os

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


try:
    from pyngrok import ngrok, conf
except ImportError:
    print("pyngrok not installed. Run: pip install pyngrok")
    sys.exit(1)

# ── 1. Set your ngrok authtoken ───────────────────────────────────────────────
# Get a free token from: https://dashboard.ngrok.com/get-started/your-authtoken
# Either set it below OR set the NGROK_AUTHTOKEN environment variable in .env
NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN", "")

if not NGROK_TOKEN:
    print("\n" + "="*60)
    print("  NGROK TOKEN NOT SET")
    print("="*60)
    print("  1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("  2. Copy your authtoken")
    print("  3. Add it to your .env file:")
    print("     NGROK_AUTHTOKEN=your_token_here")
    print("  4. Re-run: python tunnel.py")
    print("="*60 + "\n")
    sys.exit(1)

conf.get_default().auth_token = NGROK_TOKEN

# ── 2. Start Streamlit in background ─────────────────────────────────────────
print(">> Starting Streamlit...")
streamlit_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "main.py",
     "--server.headless", "true",
     "--server.port", "8501"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# Give Streamlit a moment to start
time.sleep(4)

# ── 3. Open ngrok tunnel ──────────────────────────────────────────────────────
print(">> Opening ngrok tunnel...")
tunnel = ngrok.connect(8501, "http")

print("\n" + "="*60)
print("  [OK] ECHOTALES IS LIVE!")
print("="*60)
print(f"  --> Public URL: {tunnel.public_url}")
print("="*60)
print("\n  Share this link with anyone to demo your app.")
print("  Press Ctrl+C to stop.\n")

try:
    streamlit_proc.wait()
except KeyboardInterrupt:
    print("\n>> Shutting down...")
    ngrok.disconnect(tunnel.public_url)
    ngrok.kill()
    streamlit_proc.terminate()
    print("✅ Done.")
