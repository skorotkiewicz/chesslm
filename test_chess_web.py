"""HTTP and rule checks. No data generation or training."""
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
from threading import Thread
from zipfile import ZipFile
import unittest
from unittest.mock import patch

import chess

from web.build_pages import build
from web.chess_position import position, replay
from web.chess_web import ChessServer


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # HTTP tests stub only search; rules and the HTTP server are real.
        cls.server = ChessServer(("127.0.0.1", 0), model=None, depth=1, max_nodes=100)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method="POST", path="/api/position", data=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        body = json.dumps(data) if data is not None else None
        try:
            connection.request(method, path, body, headers or {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read(), response.getheaders()
        finally:
            connection.close()

    def test_static_build_keeps_checkpoint_and_bundles_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build(output, Path("model.npz"))
            self.assertEqual((output / "model.npz").read_bytes(), Path("model.npz").read_bytes())
            self.assertIn('name="chess-runtime" content="browser"', (output / "index.html").read_text())
            self.assertTrue((output / "chess_worker.js").is_file())
            self.assertTrue((output / "neko.js").is_file())
            self.assertTrue((output / "chess_position.py").is_file())
            with ZipFile(output / "chess.zip") as archive:
                self.assertIn("chess/__init__.py", archive.namelist())
                self.assertIn("chess/engine.py", archive.namelist())
                self.assertIn("chess/LICENSE.txt", archive.namelist())
            self.assertFalse((output / "positions.jsonl").exists())

    def test_page_and_private_files(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'<div id="board"', body)
        self.assertIn(b'neko.js', body)
        self.assertIn("Content-Security-Policy", dict(headers))
        status, body, headers = self.request("GET", "/neko.js")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/javascript"))
        self.assertIn(b"chases your cursor", body)
        for path in ("/model.npz", "/../chesslm.py", "/positions.jsonl", "/chess_web.py", "/chess.zip", "/nope.js"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_legal_state_and_ai_reply(self):
        with patch("web.chess_web.choose_move", return_value=(chess.Move.from_uci("e7e5"), 31)) as search:
            status, body, _ = self.request(data={"moves": ["e2e4"], "think": True})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["moves"], ["e2e4", "e7e5"])
        self.assertEqual(result["nodes"], 31)
        self.assertEqual(result["turn"], "white")
        self.assertIn("g1f3", result["legal"])
        search.assert_called_once()
        # No shared board: another tab can still start at the initial position.
        status, body, _ = self.request(data={"moves": []})
        self.assertEqual(json.loads(body)["moves"], [])

    def test_request_validation(self):
        invalid = [None, [], {}, {"moves": "e2e4"}, {"moves": ["e2e5"]},
                   {"moves": ["0000"]}, {"moves": [42]}, {"moves": [] , "think": "yes"},
                   {"moves": ["e2e4"] * 1001}]
        for data in invalid:
            with self.subTest(data=str(data)[:80]):
                self.assertEqual(self.request(data=data)[0], 400)
        self.assertEqual(self.request(data={"moves": []}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request(data={"moves": []}, headers={"Content-Type": "application/json", "Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.request("GET", "/", headers={"Host": "example.com"})[0], 403)
        self.assertEqual(self.request(data="x" * 17000)[0], 400)

    def test_busy_and_finished_game(self):
        with self.server.search_slot:
            self.assertEqual(self.request(data={"moves": [], "think": True})[0], 503)
        moves = "f2f3 e7e5 g2g4 d8h4".split()
        with patch("web.chess_web.choose_move", side_effect=AssertionError("Finished game searched")):
            status, body, _ = self.request(data={"moves": moves, "think": True})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertTrue(result["over"])
        self.assertEqual(result["result"], "0-1")
        self.assertEqual(result["legal"], [])
        with self.assertRaises(ValueError):
            replay(moves + ["a2a3"])

    def test_special_moves_and_repetition(self):
        board = replay("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1".split())
        self.assertEqual(board.piece_at(chess.F1).symbol(), "R")
        board = replay("e2e4 a7a6 e4e5 d7d5 e5d6".split())
        self.assertIsNone(board.piece_at(chess.D5))
        board = replay("a2a4 h7h5 a4a5 h5h4 a5a6 h4h3 a6b7 h3g2 b7a8n".split())
        self.assertEqual(board.piece_at(chess.A8).symbol(), "N")
        result = position(replay("g1f3 g8f6 f3g1 f6g8".split() * 4))
        self.assertTrue(result["over"])
        self.assertEqual(result["reason"], "fivefold repetition")


if __name__ == "__main__":
    unittest.main()
