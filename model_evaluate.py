"""
Ask a chesslm checkpoint for one move from UCI move history.

   python model_evaluate.py e2e4 --depth 3 --max-nodes 20000
   python model_evaluate.py --model model.npz
"""
import argparse
import json
from pathlib import Path
from zipfile import BadZipFile

from chesslm import Model, choose_move, positive
from web.chess_position import position, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moves", nargs="*", metavar="MOVE", help="UCI moves from the starting position")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument("--depth", type=positive, default=3)
    parser.add_argument("--max-nodes", type=positive, default=20000)
    parser.add_argument("--quiescence-depth", type=positive, default=4)
    args = parser.parse_args()
    try:
        board = replay(args.moves)
        nodes = 0
        if not board.is_game_over():
            move, nodes = choose_move(board, Model.load(args.model), args.depth, args.max_nodes,
                                      quiescence_depth=args.quiescence_depth)
            board.push(move)
        print(json.dumps(position(board, nodes)))
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
