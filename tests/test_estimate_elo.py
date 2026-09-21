"""Rating math and match rules, without training or an engine process."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import chess

from estimate_elo import OPENINGS, play_game, summarize
from web.chess_position import replay


class EloTests(unittest.TestCase):
    def test_rating_and_uncertainty(self):
        even = summarize(["win", "draw", "loss", "draw"], 1320, 4)
        self.assertEqual(even["score_fraction"], 0.5)
        self.assertEqual(even["nominal_elo_estimate"], 1320)
        self.assertEqual(even["nominal_elo_95_bounds"], [None, None])
        winning = summarize(["win"] * 3 + ["loss"], 1320, 4)
        self.assertAlmostEqual(winning["nominal_elo_estimate"], 1510.8)
        few = summarize(["win", "loss"] * 20, 1320, 40)["nominal_elo_95_bounds"]
        many = summarize(["win", "loss"] * 100, 1320, 200)["nominal_elo_95_bounds"]
        self.assertLess(few[0], many[0])
        self.assertLess(many[0], 1320)
        self.assertGreater(few[1], many[1])
        self.assertGreater(many[1], 1320)

    def test_unbounded_and_incomplete_results(self):
        wins = summarize(["win"] * 40, 1320, 40)
        self.assertEqual(wins["status"], "unbounded_above")
        self.assertIsNone(wins["nominal_elo_estimate"])
        self.assertIsNone(wins["nominal_elo_95_bounds"][1])
        self.assertGreater(wins["nominal_elo_95_bounds"][0], 1320)
        losses = summarize(["loss"] * 40, 1320, 40)
        self.assertEqual(losses["status"], "unbounded_below")
        self.assertIsNone(losses["nominal_elo_estimate"])
        self.assertIsNone(losses["nominal_elo_95_bounds"][0])
        self.assertLess(losses["nominal_elo_95_bounds"][1], 1320)
        for results in ([], ["win"], ["win", "unfinished"]):
            report = summarize(results, 1320, 2)
            self.assertEqual(report["status"], "incomplete")
            self.assertIsNone(report["nominal_elo_estimate"])
            self.assertIsNone(report["nominal_elo_95_bounds"])

    def test_games(self):
        args = SimpleNamespace(depth=1, max_nodes=100, quiescence_depth=8, max_plies=4, sf_time=0.01)
        mate = chess.Move.from_uci("d8h4")
        for color, result in ((chess.WHITE, "loss"), (chess.BLACK, "win")):
            engine = Mock()
            engine.play.return_value = SimpleNamespace(move=mate)
            with patch("estimate_elo.choose_move", return_value=(mate, 12)) as search:
                game = play_game(None, engine, "f2f3 e7e5 g2g4", color, args)
            self.assertEqual(game["model_result"], result)
            self.assertEqual(game["result"], "0-1")
            self.assertEqual(game["termination"], "checkmate")
            self.assertEqual(game["plies"], 4)  # Mate on the last allowed ply still counts.
            self.assertTrue(replay(game["moves"]).is_checkmate())
            if color == chess.BLACK:
                self.assertEqual(search.call_args.kwargs, {"quiescence_depth": 8})
                engine.play.assert_not_called()
            else:
                search.assert_not_called()
                engine.play.assert_called_once()
        for opening in OPENINGS:
            self.assertEqual(replay(opening.split()).turn, chess.WHITE)
            game = play_game(None, Mock(), opening, chess.WHITE, args)
            self.assertEqual(game["model_result"], "unfinished")
            self.assertEqual(game["result"], "*")
        opening = "g1f3 g8f6 f3g1 f6g8 " * 2
        game = play_game(None, Mock(), opening, chess.WHITE, args)
        self.assertEqual(game["model_result"], "draw")
        self.assertEqual(game["termination"], "threefold_repetition")
        engine = Mock()
        engine.play.return_value = SimpleNamespace(move=chess.Move.from_uci("a1a8"))
        with self.assertRaisesRegex(ValueError, "illegal move"):
            play_game(None, engine, "f2f3 e7e5 g2g4", chess.WHITE, args)


if __name__ == "__main__":
    unittest.main()
