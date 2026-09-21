"""Standalone model evaluator CLI check."""
import json
import subprocess
import sys
import unittest

from web.chess_position import replay


class ModelEvaluateTests(unittest.TestCase):
    def test_model_move_from_history(self):
        result = subprocess.run(
            [sys.executable, "model_evaluate.py", "e2e4", "--depth", "1", "--max-nodes", "100"],
            check=True, capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        self.assertEqual(data["moves"][0], "e2e4")
        self.assertEqual(len(replay(data["moves"]).move_stack), 2)
        self.assertLessEqual(data["nodes"], 100)


if __name__ == "__main__":
    unittest.main()
