"""Loopback-only UI; no microphone capture, cloud calls or arbitrary tool execution."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from .core import POLICIES, evaluate
from .evaluation import run_suite
from .runtime import doctor
from .scenarios import FAMILIES, generate

ROOT = Path(__file__).resolve().parent.parent
RUN_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "STSLab/0.1"

    def allowed_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def send(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.send_header("Permissions-Policy", "microphone=(), camera=()")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed_host():
            return self.send(403, {"error": "invalid host"})
        url = urlsplit(self.path)
        if url.path == "/api/report":
            path = ROOT / "reports/held_out.json"
            return self.send(200, json.loads(path.read_text())) if path.exists() else self.send(404, {"error": "Run the evaluation first"})
        if url.path == "/api/status":
            return self.send(200, doctor())
        if url.path == "/api/native-report":
            path = ROOT / "reports/native-audio.json"
            return self.send(200, json.loads(path.read_text())) if path.exists() else self.send(404, {"error": "Native probe has not run"})
        if url.path == "/api/scenarios":
            return self.send(200, {"families": FAMILIES, "variants": list(POLICIES)})
        if url.path == "/api/document":
            name = parse_qs(url.query).get("name", [""])[0]
            if name not in ("research", "build-memo", "evaluation"):
                return self.send(404, {"error": "unknown document"})
            path = ROOT / "docs" / f"{name}.md"
            if not path.exists():
                return self.send(404, {"error": "document unavailable"})
            return self.send(200, {"text": path.read_text()})
        files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
        if url.path in files:
            name, content_type = files[url.path]
            return self.send(200, (ROOT / "web" / name).read_bytes(), content_type+"; charset=utf-8")
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.allowed_host():
            return self.send(403, {"error": "invalid host"})
        allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if self.headers.get("Origin") not in allowed:
            return self.send(403, {"error": "local browser origin required"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.send(415, {"error": "application/json required"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                return self.send(413, {"error": "request size out of bounds"})
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("object required")
        except (ValueError, UnicodeDecodeError):
            return self.send(400, {"error": "invalid JSON request"})
        if self.path == "/api/evaluate":
            if not RUN_LOCK.acquire(blocking=False):
                return self.send(409, {"error": "evaluation already running"})
            try:
                return self.send(200, run_suite(output=ROOT / "reports"))
            finally:
                RUN_LOCK.release()
        if self.path == "/api/replay":
            family, variant, index = payload.get("family"), payload.get("variant"), payload.get("index", 0)
            if family not in FAMILIES or variant not in POLICIES or type(index) is not int or not 0 <= index < 20:
                return self.send(400, {"error": "invalid family, variant or scenario index"})
            scenario = [s for s in generate(seed=0) if s.family == family][index]
            return self.send(200, evaluate(scenario, POLICIES[variant]))
        return self.send(404, {"error": "not found"})

    def log_message(self, *_):
        pass


def create_server(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def serve(port=8765):
    server = create_server(port)
    print(f"STS Reliability Lab: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
