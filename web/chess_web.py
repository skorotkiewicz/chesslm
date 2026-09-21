# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0,<3", "python-chess==1.999"]
# ///
"""Local browser chess using model.npz. No training or Stockfish."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import BoundedSemaphore
import traceback
from zipfile import BadZipFile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import chess
import chess.engine
import sys

# Running from the repository root needs the parent directory for chesslm.py;
# sibling imports need this directory.
sys.path.extend((str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)))
from chesslm import ENGINE, Model, choose_move, positive
from chess_position import position, replay

PAGE = Path(__file__).with_name("chess_web.html")
# Same-origin assets the page loads besides itself, with their real types.
ASSETS = {"/neko.js": ("neko.js", "application/javascript"),
          "/favicon.svg": ("favicon.svg", "image/svg+xml")}


class ChessServer(ThreadingHTTPServer):
    def __init__(self, address, model, depth=3, max_nodes=20000, engine=ENGINE, sf_nodes=10000,
                 quiescence_depth=4):
        self.model, self.depth, self.max_nodes = model, depth, max_nodes
        self.quiescence_depth = quiescence_depth
        self.engine_path, self.sf_nodes = engine, sf_nodes
        # ponytail: one CPU search at a time; use a bounded worker pool for multi-user hosting.
        self.search_slot = BoundedSemaphore(1)
        self.page = PAGE.read_bytes()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def send(self, status, data, content_type="application/json"):
        body = json.dumps(data).encode() if content_type == "application/json" else data
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        port = self.server.server_port
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host not in (f"localhost:{port}", f"127.0.0.1:{port}") or origin not in (None, f"http://{host}"):
            self.send(403, {"error": "Only same-origin localhost requests are allowed"})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        asset = ASSETS.get(self.path.split("?")[0])
        if self.path == "/":
            self.send(200, self.server.page, "text/html")
        elif asset:
            try:
                self.send(200, (PAGE.parent / asset[0]).read_bytes(), asset[1])
            except OSError:
                self.send(404, {"error": "Not found"})
        else:
            self.send(404, {"error": "Not found"})

    def do_POST(self):
        if not self.local_request():
            return
        if self.path != "/api/position":
            self.send(404, {"error": "Not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send(415, {"error": "Use application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or not 0 < length <= 16384:
                raise ValueError("Request body must contain 1 to 16384 bytes")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or type(data.get("think", False)) is not bool \
                    or type(data.get("watch", False)) is not bool:
                raise ValueError("Expected an object with moves and boolean think and watch flags")
            board = replay(data.get("moves"))
        except (ValueError, UnicodeError, TimeoutError) as error:
            self.send(400, {"error": str(error)})
            return
        nodes = 0
        if (data.get("think") or data.get("watch")) and not board.is_game_over():
            if not self.server.search_slot.acquire(blocking=False):
                self.send(503, {"error": "Model is busy in another tab. Try again shortly."})
                return
            try:
                move, nodes = choose_move(board, self.server.model, self.server.depth, self.server.max_nodes,
                                          quiescence_depth=self.server.quiescence_depth)
                if move not in board.legal_moves:
                    raise ValueError("Search returned an illegal move")
                board.push(move)
                # Watch mode: the model moves, then Stockfish answers in the same request.
                if data.get("watch") and not board.is_game_over() and not self.engine_move(board):
                    return
            except Exception:
                traceback.print_exc()
                self.send(500, {"error": "Search failed. Check the server terminal."})
                return
            finally:
                self.server.search_slot.release()
        self.send(200, position(board, nodes))

    def engine_move(self, board):
        try:
            with chess.engine.SimpleEngine.popen_uci(str(self.server.engine_path)) as engine:
                engine.configure({"Threads": 1, "Hash": 16})
                result = engine.play(board, chess.engine.Limit(nodes=self.server.sf_nodes))
        except (OSError, chess.engine.EngineError):
            self.send(503, {"error": "Watch mode needs a working Stockfish; pass --engine /path/to/stockfish"})
            return False
        board.push(result.move)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent.parent / "model.npz")
    parser.add_argument("--port", type=positive, default=8000)
    parser.add_argument("--depth", type=positive, default=3)
    parser.add_argument("--max-nodes", type=positive, default=20000)
    parser.add_argument("--quiescence-depth", type=positive, default=4,
                        help="Maximum extra tactical plies, within --max-nodes (default: 4)")
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--sf-nodes", type=positive, default=10000)
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("Port must not exceed 65535")
    try:
        model = Model.load(args.model)
        with ChessServer(("127.0.0.1", args.port), model, args.depth, args.max_nodes,
                         args.engine, args.sf_nodes, quiescence_depth=args.quiescence_depth) as server:
            print(f"Open http://127.0.0.1:{args.port} | model: {args.model.name}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        parser.exit(1, f"Cannot start chess server: {error}\n")


if __name__ == "__main__":
    main()
