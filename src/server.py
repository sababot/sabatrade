import http.server
import socketserver
import subprocess
import threading
import time
import json
import os
import sys
import signal

PORT = 8000
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "real.py")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Global state
LOGS = []
LOG_LOCK = threading.Lock()
PROCESS = None

def run_trading_bot():
    global PROCESS
    # Force color output for rich and unbuffer python output
    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = [sys.executable, "-u", SCRIPT_PATH]
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    print(f"Server: Starting bot from {project_root}...")
    
    try:
        # Popen with stdout piped
        PROCESS = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1, 
            env=env
        )
        
        # Read stdout line by line
        for line in iter(PROCESS.stdout.readline, ''):
            if not line:
                break
            with LOG_LOCK:
                LOGS.append(line)
                # Keep memory usage in check, keep last 20k lines.
                if len(LOGS) > 20000:
                    LOGS.pop(0)
                    
        PROCESS.stdout.close()
        return_code = PROCESS.wait()
        with LOG_LOCK:
            LOGS.append(f"\n[Server] Process exited with code {return_code}\n")
            
    except Exception as e:
        with LOG_LOCK:
            LOGS.append(f"\n[Server] Error running process: {e}\n")


class BotRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            try:
                with open(os.path.join(TEMPLATE_DIR, "index.html"), 'rb') as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.wfile.write(f"Error loading template: {e}".encode())
            return

        if self.path.startswith('/logs'):
            # Parse query params manually
            query_part = self.path.split('?')[-1] if '?' in self.path else ''
            params = {}
            if query_part:
                for pair in query_part.split('&'):
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        params[k] = v
            
            try:
                since_index = int(params.get('since', 0))
            except ValueError:
                since_index = 0
            
            with LOG_LOCK:
                # If client requests index larger than current length, likely reset
                if since_index > len(LOGS):
                    since_index = 0
                
                # Fetch new logs
                new_logs = LOGS[since_index:]
                next_index = len(LOGS)
            
            response = {
                "logs": new_logs,
                "next_index": next_index
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return
            
        # 404 for anything else
        self.send_error(404)

def start_server():
    # Start bot thread
    print("Server: Launching bot process...")
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()
    
    # Start HTTP server
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), BotRequestHandler) as httpd:
        print(f"\n[Server] Web interface running at http://localhost:{PORT}")
        print("[Server] Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down...")
            if PROCESS:
                PROCESS.terminate()
            httpd.shutdown()
            sys.exit(0)

if __name__ == "__main__":
    start_server()
