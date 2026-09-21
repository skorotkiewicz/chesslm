"""Small GUI regression check. Requires a desktop display; never trains."""
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import patch

import chess

from chess_game import BOARD_SIZE, ChessGame


class GameTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.addCleanup(self.root.destroy)
        self.root.withdraw()
        worker = patch("chess_game.Thread")
        self.worker = worker.start()
        self.addCleanup(worker.stop)
        self.game = ChessGame(self.root, model=None, quiescence_depth=8)

    def click(self, square):
        x, y = self.game.square_center(square)
        self.game.on_click(SimpleNamespace(x=x, y=y))

    def test_coordinates_clicks_and_background_reply(self):
        game = self.game
        for color in (chess.WHITE, chess.BLACK):
            game.human = color
            for square in chess.SQUARES:
                self.assertEqual(game.square_at(*game.square_center(square)), square)
        game.human = chess.WHITE
        self.assertIsNone(game.square_at(0, 0))
        self.assertIsNone(game.square_at(BOARD_SIZE, BOARD_SIZE))
        self.click(chess.E2)
        self.click(chess.E5)  # Illegal move leaves the board unchanged.
        self.assertEqual(game.board.fen(), chess.STARTING_FEN)
        self.click(chess.E2)
        self.click(chess.E4)
        self.assertEqual(game.board.peek().uci(), "e2e4")
        self.assertTrue(game.thinking)
        self.worker.assert_called_once()
        snapshot = self.worker.call_args.kwargs["args"][0]
        self.assertIsNot(snapshot, game.board)
        self.assertEqual(snapshot.move_stack, game.board.move_stack)
        game.reset()  # Cannot reset underneath an active search.
        self.assertEqual(len(game.board.move_stack), 1)
        game.results.put((chess.Move.from_uci("e7e5"), None))
        game.poll()
        self.assertFalse(game.thinking)
        self.assertEqual(len(game.board.move_stack), 2)
        self.assertIn("Your turn", game.status.get())
        game.reset()
        self.assertEqual(game.board.fen(), chess.STARTING_FEN)

    def test_promotion_and_terminal_status(self):
        game = self.game
        game.board = chess.Board("7k/P7/6K1/8/8/8/8/8 w - - 0 1")
        self.click(chess.A7)
        self.click(chess.A8)
        self.assertEqual(len(game.pending), 4)
        self.assertEqual(len(game.board.move_stack), 0)
        game.promote(chess.KNIGHT)
        self.assertEqual(game.board.piece_type_at(chess.A8), chess.KNIGHT)
        self.assertIn("Draw", game.status.get())
        self.assertFalse(game.thinking)
        game.board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        game.refresh()
        self.assertIn("You win: checkmate", game.status.get())

    def test_black_start_and_worker_error(self):
        game = self.game
        game.human = chess.BLACK
        game.reset()
        self.assertTrue(game.thinking)
        self.click(chess.E7)
        self.assertIsNone(game.selected)
        game.results.put((None, "test failure"))
        game.poll()
        self.assertFalse(game.thinking)
        self.assertIn("AI error: test failure", game.status.get())
        self.assertEqual(str(game.restart["state"]), "normal")

    def test_search_quiescence_depth(self):
        board = self.game.board.copy()
        move = chess.Move.from_uci("e2e4")
        with patch("chess_game.choose_move", return_value=(move, 10)) as search:
            self.game.search(board)
        search.assert_called_once_with(board, None, 3, 20000, quiescence_depth=8)
        self.assertEqual(self.game.results.get_nowait(), (move, None))

    def test_keyboard_selection(self):
        self.game.on_key(SimpleNamespace(keysym="Return"))
        self.assertEqual(self.game.selected, chess.E2)
        self.game.on_key(SimpleNamespace(keysym="Up"))
        self.game.on_key(SimpleNamespace(keysym="Return"))
        self.assertEqual(self.game.board.peek().uci(), "e2e3")


if __name__ == "__main__":
    unittest.main()
