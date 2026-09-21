"""
Estimate nominal Elo against strength-limited Stockfish, not a human rating.

   python estimate_elo.py --model model.npz --games 40 \
     --opponent-elo 1320 --sf-time 0.1 \
     --depth 3 --max-nodes 20000 --quiescence-depth 4
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from zipfile import BadZipFile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import chess
import chess.engine

from chesslm import ENGINE, Model, choose_move, positive
from web.chess_position import replay

# ponytail: small opening set; use a larger independent suite for stronger rating claims.
OPENINGS = (
    "e2e4 e7e5 g1f3 b8c6",
    "d2d4 d7d5 c2c4 e7e6",
    "e2e4 c7c5 g1f3 d7d6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "d2d4 g8f6 c2c4 g7g6",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 g2g3 g8f6",
)
WARNING = (
    "Nominal estimate anchored to Stockfish's UCI_Elo setting. That setting is not "
    "calibrated for this time budget or hardware. Model and Stockfish budgets differ. "
    "This is not a FIDE, Chess.com, or Lichess rating."
)


def summarize(results, opponent_elo, requested_games):
    counts = Counter(results)
    report = {"type": "summary", "games_requested": requested_games, "games_played": len(results),
              "wins": counts["win"], "draws": counts["draw"], "losses": counts["loss"],
              "unfinished": counts["unfinished"], "opponent_elo_setting": opponent_elo,
              "score_fraction": None, "nominal_elo_estimate": None,
              "nominal_elo_95_bounds": None, "status": "incomplete", "warning": WARNING}
    # Dropping unfinished games could bias the estimate, so require the full schedule.
    if len(results) != requested_games or counts["unfinished"] or not results or len(results) % 2:
        return report
    score = (counts["win"] + counts["draw"] / 2) / len(results)

    def elo(fraction):
        if fraction <= 0 or fraction >= 1:
            return None  # Unbounded, not an invented finite rating.
        return round(opponent_elo + 400 * math.log10(fraction / (1 - fraction)), 1)

    # Hoeffding's bound uses independent pair scores in [0, 1], including draws.
    # Pairing avoids treating the two colors of one opening as independent samples.
    radius = math.sqrt(math.log(2 / 0.05) / (2 * (len(results) // 2)))
    report.update(score_fraction=score, nominal_elo_estimate=elo(score),
                  nominal_elo_95_bounds=[elo(max(0, score - radius)), elo(min(1, score + radius))],
                  status="unbounded_below" if score == 0 else "unbounded_above" if score == 1 else "estimated",
                  interval_method="95% Hoeffding bounds assuming independent opening pairs; null endpoints are unbounded")
    return report


def play_game(model, engine, opening, model_color, args):
    board = replay(opening.split())
    game_id = object()  # Reset Stockfish's game state between games.
    started = time.monotonic()
    nodes = 0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < args.max_plies:
        if board.turn == model_color:
            move, searched = choose_move(board, model, args.depth, args.max_nodes,
                                         quiescence_depth=args.quiescence_depth)
            nodes += searched
        else:
            move = engine.play(board, chess.engine.Limit(time=args.sf_time), game=game_id).move
        if move not in board.legal_moves:
            raise ValueError("Player returned an illegal move")
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    result = "unfinished" if outcome is None else "draw" if outcome.winner is None else (
        "win" if outcome.winner == model_color else "loss")
    return {"type": "game", "opening": opening, "model_color": "white" if model_color else "black",
            "model_result": result, "result": outcome.result() if outcome else "*",
            "termination": outcome.termination.name.lower() if outcome else "ply_limit",
            "plies": len(board.move_stack), "model_nodes": nodes,
            "seconds": round(time.monotonic() - started, 3),
            "moves": [move.uci() for move in board.move_stack]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--games", type=positive, default=40, help="Even number of games (default: 40)")
    parser.add_argument("--opponent-elo", type=positive, default=1320, help="Stockfish UCI_Elo setting")
    parser.add_argument("--sf-time", type=float, default=0.1, help="Stockfish seconds per move (default: 0.1)")
    parser.add_argument("--depth", type=positive, default=3)
    parser.add_argument("--max-nodes", type=positive, default=20000)
    parser.add_argument("--quiescence-depth", type=positive, default=4)
    parser.add_argument("--max-plies", type=positive, default=400, help="Game limit including opening plies")
    parser.add_argument("--seed", type=int, default=42, help="Opening schedule seed, not Stockfish's internal seed")
    args = parser.parse_args()
    if args.games % 2:
        parser.error("--games must be even so every opening is played with both colors")
    if not math.isfinite(args.sf_time) or args.sf_time <= 0:
        parser.error("--sf-time must be finite and positive")
    results = []
    rng = random.Random(args.seed)
    interrupted = False
    try:
        model = Model.load(args.model)
        with chess.engine.SimpleEngine.popen_uci(str(args.engine)) as engine:
            option = engine.options.get("UCI_Elo")
            if option is None or "UCI_LimitStrength" not in engine.options:
                raise ValueError("Engine must support UCI_Elo and UCI_LimitStrength")
            if not option.min <= args.opponent_elo <= option.max:
                raise ValueError(f"--opponent-elo must be between {option.min} and {option.max}")
            engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": args.opponent_elo})
            print(json.dumps({"type": "settings", "engine": engine.id, "settings": vars(args),
                              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                              "engine_threads": 1, "engine_hash_mb": 64, "claim_draw": True,
                              "warning": WARNING}, default=str), flush=True)
            try:
                for pair in range(args.games // 2):
                    opening = rng.choice(OPENINGS)
                    for color in (chess.WHITE, chess.BLACK):
                        game = play_game(model, engine, opening, color, args)
                        results.append(game["model_result"])
                        print(json.dumps(dict(game, game=len(results), pair=pair + 1)), flush=True)
            except KeyboardInterrupt:
                interrupted = True
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, chess.engine.EngineError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(summarize(results, args.opponent_elo, args.games)), flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
