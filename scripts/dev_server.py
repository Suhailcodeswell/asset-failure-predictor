"""Local dev server: static UI + predict API (no Node/Vercel required)."""
import importlib.util
import mimetypes
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
API_DIR = ROOT / "api"

sys.path.insert(0, str(API_DIR))
spec = importlib.util.spec_from_file_location("predict", API_DIR / "predict.py")
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


class DevHandler(predict.handler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/api") or "/auth" in self.path:
            return super().do_GET()
        return self._serve_static(path)

    def _serve_static(self, path: str):
        if path in ("/", ""):
            rel = "index.html"
        else:
            rel = unquote(path.lstrip("/"))
        file_path = (PUBLIC / rel).resolve()
        if not str(file_path).startswith(str(PUBLIC.resolve())) or not file_path.is_file():
            self.send_error(404)
            return
        content = file_path.read_bytes()
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        print(f"[dev] {self.address_string()} - {fmt % args}")


def main():
    port = int(os.environ.get("PORT", "8765"))
    server = HTTPServer(("127.0.0.1", port), DevHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"Asset Failure Predictor dev server: {url}")
    print("Access code: AFPredict2026!")
    server.serve_forever()


if __name__ == "__main__":
    main()
