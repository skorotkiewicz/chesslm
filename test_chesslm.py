"""Forward-only checks. No Stockfish process, data generation, or training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess
import numpy as np

from chesslm import (HIDDEN, INPUTS, Model, board_from_fen, choose_move,
                     features, material, split_rows)


class ChessTests(unittest.TestCase):
    def setUp(self):
        # Hand-set zero weights test the material baseline, not a trained model.
        self.model = Model(np.zeros((INPUTS, HIDDEN), np.float32),
                           np.zeros(HIDDEN, np.float32),
                           np.zeros(HIDDEN, np.float32), np.zeros(1, np.float32))

    def test_size_and_encoding(self):
        self.assertEqual(sum(p.size for p in self.model.params), 100353)
        self.assertLess(sum(p.nbytes for p in self.model.params), 500000)
        board = chess.Board()
        x = features(board)
        self.assertEqual(x[:768].sum(), 32)
        np.testing.assert_array_equal(x[769:773], [1, 1, 1, 1])
        self.assertEqual(material(x), 0)
        self.assertEqual(float(self.model.predict(x)), 0)
        board.push_uci("e2e4")
        self.assertEqual(features(board)[768], -1)

    def test_special_features(self):
        board = chess.Board()
        for move in ("e2e4", "a7a6", "e4e5", "d7d5"):
            board.push_uci(move)
        self.assertEqual(features(board)[773 + 3], 1)
        board.halfmove_clock = 75
        self.assertEqual(features(board)[781], 0.5)
        self.assertAlmostEqual(float(material(features(board))),
                               -float(material(features(board.mirror()))))

    def test_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.npz"
            self.model.save(path)
            self.assertLess(path.stat().st_size, 500000)
            loaded = Model.load(path)
            np.testing.assert_array_equal(loaded.predict(features(chess.Board())),
                                          self.model.predict(features(chess.Board())))
            with self.assertRaises(FileExistsError):
                self.model.save(path)
            invalid = Path(directory) / "bad.npz"
            self.model.params[0][0, 0] = np.nan
            self.model.save(invalid)
            with self.assertRaises(ValueError):
                Model.load(invalid)

    def test_legal_move_and_budget_without_engine(self):
        board = chess.Board()
        before = board.fen()
        with patch("chess.engine.SimpleEngine.popen_uci", side_effect=AssertionError("Engine called")):
            for budget in (1, 100):
                move, nodes = choose_move(board, self.model, depth=3, max_nodes=budget)
                self.assertIn(move, board.legal_moves)
                self.assertLessEqual(nodes, budget)
                self.assertEqual(board.fen(), before)
                self.assertEqual(board.move_stack, [])

    def test_mate_for_both_colors(self):
        white = board_from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        for board in (white, white.mirror()):
            move, _ = choose_move(board, self.model, depth=2, max_nodes=2000)
            board.push(move)
            self.assertTrue(board.is_checkmate())

    def test_capture_free_queen(self):
        board = board_from_fen("4k3/8/8/8/8/8/4q3/4R1K1 w - - 0 1")
        move, _ = choose_move(board, self.model, depth=2, max_nodes=3000)
        self.assertEqual(move.uci(), "e1e2")

    def test_promotion_and_check_evasion(self):
        for fen in ("7k/P7/6K1/8/8/8/8/8 w - - 0 1",
                    "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"):
            board = board_from_fen(fen)
            move, _ = choose_move(board, self.model, depth=2, max_nodes=1000)
            self.assertIn(move, board.legal_moves)
            if board.piece_at(chess.A7):
                self.assertIsNotNone(move.promotion)

    def test_game_over(self):
        for fen in ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
                    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
                    "7k/8/6K1/8/8/8/8/8 w - - 0 1",
                    "7k/8/6K1/8/8/8/P7/8 w - - 150 80"):
            self.assertEqual(choose_move(board_from_fen(fen), self.model), (None, 0))

    def test_invalid_fen(self):
        with self.assertRaises(ValueError):
            board_from_fen("8/8/8/8/8/8/8/8 w - - 0 1")

    def test_game_split_removes_shared_positions(self):
        board = chess.Board()
        common = {"fen": board.fen(), "cp": 0}
        rows = []
        for game, move in enumerate(("e2e4", "d2d4", "c2c4")):
            board = chess.Board()
            board.push_uci(move)
            rows.extend([dict(common, game=game),
                         {"fen": board.fen(), "cp": 20, "game": game}])
        train, valid = split_rows(rows, 42)
        self.assertTrue({r["game"] for r in train}.isdisjoint({r["game"] for r in valid}))
        self.assertTrue({r["fen"] for r in train}.isdisjoint({r["fen"] for r in valid}))
        with self.assertRaises(ValueError):
            split_rows([dict(common, game=0)], 42)


if __name__ == "__main__":
    unittest.main()
