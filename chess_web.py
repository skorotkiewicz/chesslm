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
from chesslm import Model, choose_move, positive

PAGE = Path(__file__).with_name("chess_web.html")


def replay(moves):
    if not isinstance(moves, list) or len(moves) > 1000:
        raise ValueError("Expected a list of at most 1000 moves")
    board = chess.Board()
    for text in moves:
        if not isinstance(text, str) or len(text) not in (4, 5):
            raise ValueError("Moves must be UCI strings, such as e2e4")
        move = chess.Move.from_uci(text)
        if board.is_game_over() or move not in board.legal_moves:
            raise ValueError(f"Illegal move: {text}")
        board.push(move)
    return board


def position(board, nodes=0):
    outcome = board.outcome()
    return {
        "moves": [m.uci() for m in board.move_stack],
        "pieces": {chess.square_name(sq): piece.symbol() for sq, piece in board.piece_map().items()},
        "legal": [m.uci() for m in board.legal_moves] if not outcome else [],
        "turn": "white" if board.turn else "black",
        "check": chess.square_name(board.king(board.turn)) if board.is_check() else None,
        "over": outcome is not None,
        "result": outcome.result() if outcome else None,
        "reason": outcome.termination.name.replace("_", " ").lower() if outcome else None,
        "nodes": nodes,
    }


class ChessServer(ThreadingHTTPServer):
    def __init__(self, address, model, depth=3, max_nodes=20000):
        self.model, self.depth, self.max_nodes = model, depth, max_nodes
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
        if self.path == "/":
            self.send(200, self.server.page, "text/html")
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
            if not isinstance(data, dict) or type(data.get("think", False)) is not bool:
                raise ValueError("Expected an object with moves and a boolean think flag")
            board = replay(data.get("moves"))
        except (ValueError, UnicodeError, TimeoutError) as error:
            self.send(400, {"error": str(error)})
            return
        nodes = 0
        if data.get("think") and not board.is_game_over():
            if not self.server.search_slot.acquire(blocking=False):
                self.send(503, {"error": "Model is busy in another tab. Try again shortly."})
                return
            try:
                move, nodes = choose_move(board, self.server.model, self.server.depth, self.server.max_nodes)
                if move not in board.legal_moves:
                    raise ValueError("Search returned an illegal move")
                board.push(move)
            except Exception:
                traceback.print_exc()
                self.send(500, {"error": "Model search failed. Check the server terminal."})
                return
            finally:
                self.server.search_slot.release()
        self.send(200, position(board, nodes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument("--port", type=positive, default=8000)
    parser.add_argument("--depth", type=positive, default=3)
    parser.add_argument("--max-nodes", type=positive, default=20000)
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("Port must not exceed 65535")
    try:
        model = Model.load(args.model)
        with ChessServer(("127.0.0.1", args.port), model, args.depth, args.max_nodes) as server:
            print(f"Open http://127.0.0.1:{args.port} | model: {args.model.name}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        parser.exit(1, f"Cannot start chess server: {error}\n")


if __name__ == "__main__":
    main()
