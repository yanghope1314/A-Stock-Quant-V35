"""Poll port and open browser when Django server is ready."""
import time
import socket
import webbrowser

PORT = 8000
MAX_WAIT = 60  # seconds

print(f"Polling 127.0.0.1:{PORT} ...")
for i in range(MAX_WAIT // 2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            s.close()
            print("Server ready, opening browser...")
            webbrowser.open(f"http://127.0.0.1:{PORT}")
            break
    except Exception:
        pass
    s.close()
    time.sleep(2)
else:
    print(f"Server did not start within {MAX_WAIT}s. Open http://127.0.0.1:{PORT} manually.")
