"""Tiny chess evaluator distilled from Stockfish. No training runs on import."""
import argparse
import json
from pathlib import Path
import random

import chess
import chess.engine
import numpy as np

ENGINE = Path(__file__).resolve().parent / "stockfish/stockfish-linux-x86-64-universal"
INPUTS, HIDDEN = 782, 128
SCALE = 600.0
PIECE_VALUES = (0, 100, 320, 330, 500, 900, 0)
MATE = 100_000


def features(board):
    """White-oriented piece planes, turn, rights, en passant file, halfmove clock."""
    x = np.zeros(INPUTS, dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = piece.piece_type - 1 + (0 if piece.color else 6)
        x[plane * 64 + square] = 1
    x[768] = 1 if board.turn else -1
    x[769:773] = [board.has_kingside_castling_rights(chess.WHITE),
                  board.has_queenside_castling_rights(chess.WHITE),
                  board.has_kingside_castling_rights(chess.BLACK),
                  board.has_queenside_castling_rights(chess.BLACK)]
    if board.has_legal_en_passant():
        x[773 + chess.square_file(board.ep_square)] = 1
    x[781] = min(board.halfmove_clock, 150) / 150
    return x


def material(x):
    values = np.repeat([100, 320, 330, 500, 900, 0,
                        -100, -320, -330, -500, -900, 0], 64).astype(np.float32)
    return x[..., :768] @ values / SCALE


class Model:
    def __init__(self, w1, b1, w2, b2):
        self.params = [w1, b1, w2, b2]

    @classmethod
    def fresh(cls, seed):
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, 0.05, (INPUTS, HIDDEN)).astype(np.float32),
                   np.zeros(HIDDEN, np.float32),
                   rng.normal(0, 0.01, HIDDEN).astype(np.float32),
                   np.zeros(1, np.float32))

    def predict(self, x):
        w1, b1, w2, b2 = self.params
        return np.tanh(material(x) + np.tanh(x @ w1 + b1) @ w2 + b2[0])

    def gradients(self, x, target):
        w1, b1, w2, b2 = self.params
        hidden = np.tanh(x @ w1 + b1)
        pred = np.tanh(material(x) + hidden @ w2 + b2[0])
        delta = 2 * (pred - target) * (1 - pred * pred) / len(x)
        dh = delta[:, None] * w2 * (1 - hidden * hidden)
        return [x.T @ dh, dh.sum(axis=0), hidden.T @ delta,
                np.array([delta.sum()], dtype=np.float32)]

    def save(self, path):
        # Exclusive creation prevents overwriting a previous checkpoint.
        with open(path, "xb") as stream:
            np.savez(stream, version=np.array(1),
                     **dict(zip(("w1", "b1", "w2", "b2"), self.params)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if data["version"].shape != () or data["version"].item() != 1:
                raise ValueError("Unsupported model version")
            params = [data[key] for key in ("w1", "b1", "w2", "b2")]
        for param, shape in zip(params, ((INPUTS, HIDDEN), (HIDDEN,), (HIDDEN,), (1,))):
            if param.shape != shape or param.dtype != np.float32 or not np.isfinite(param).all():
                raise ValueError("Invalid model weights")
        return cls(*params)


def board_from_fen(fen):
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("Invalid chess position")
    return board


def generate(args):
    """Bounded Stockfish self-play with varied opening choices."""
    rng = random.Random(args.seed)
    count, game = 0, 0
    with open(args.output, "x") as out, chess.engine.SimpleEngine.popen_uci(str(args.engine)) as engine:
        engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        while count < args.positions:
            board = chess.Board()
            for ply in range(240):
                if board.is_game_over(claim_draw=True) or count >= args.positions:
                    break
                lines = engine.analyse(board, chess.engine.Limit(nodes=args.nodes), multipv=3)
                score = lines[0]["score"].white().score(mate_score=10000)
                out.write(json.dumps({"fen": board.fen(), "cp": score, "game": game}) + "\n")
                count += 1
                if count % 100 == 0:
                    out.flush()
                    print(f"positions={count}/{args.positions}", flush=True)
                choice = rng.choice(lines) if ply < 16 or rng.random() < 0.1 else lines[0]
                board.push(choice["pv"][0])
            game += 1
    print(f"Saved {count} positions from {game} games to {args.output}")


def load_rows(path):
    rows = []
    with open(path) as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                board_from_fen(row["fen"])
                if not isinstance(row["game"], int) or not np.isfinite(float(row["cp"])):
                    raise ValueError("Invalid game or score")
                rows.append(row)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid dataset row {number}: {error}") from error
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def position_key(fen):
    return " ".join(fen.split()[:4])


def split_rows(rows, seed):
    games = sorted({row["game"] for row in rows})
    if len(games) < 2:
        raise ValueError("Need at least two games for held-out validation")
    random.Random(seed).shuffle(games)
    held_out = set(games[:max(1, len(games) // 10)])
    valid = [row for row in rows if row["game"] in held_out]
    seen = {position_key(row["fen"]) for row in valid}
    train = [row for row in rows if row["game"] not in held_out
             and position_key(row["fen"]) not in seen]
    if not train:
        raise ValueError("No training positions remain after validation deduplication")
    return train, valid


def arrays(rows):
    return (np.stack([features(board_from_fen(row["fen"])) for row in rows]),
            np.tanh(np.array([float(row["cp"]) for row in rows], np.float32) / SCALE))


def train(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    training, validation = split_rows(load_rows(args.data), args.seed)
    # ponytail: dense arrays fit small datasets; stream batches for millions of positions.
    x, y = arrays(training)
    vx, vy = arrays(validation)
    model = Model.fresh(args.seed)
    rng = np.random.default_rng(args.seed)
    first = [np.zeros_like(p) for p in model.params]
    second = [np.zeros_like(p) for p in model.params]

    def validation_loss():
        total = 0.0
        for start in range(0, len(vx), args.batch_size):
            pred = model.predict(vx[start:start + args.batch_size])
            total += float(np.square(pred - vy[start:start + args.batch_size]).sum())
        return total / len(vx)

    best_loss = validation_loss()
    best = [p.copy() for p in model.params]
    print(f"train={len(x)} validation={len(vx)} initial_mse={best_loss:.6f}", flush=True)
    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), args.batch_size):
            batch = order[start:start + args.batch_size]
            gradients = model.gradients(x[batch], y[batch])
            step += 1
            for p, m, v, g in zip(model.params, first, second, gradients):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                p -= args.learning_rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        loss = validation_loss()
        if not np.isfinite(loss):
            raise ValueError("Non-finite validation loss; no checkpoint saved")
        if loss < best_loss:
            best_loss, best = loss, [p.copy() for p in model.params]
        print(f"epoch={epoch + 1} validation_mse={loss:.6f} best={best_loss:.6f}", flush=True)
    Model(*best).save(args.output)
    print(f"Saved best checkpoint to {args.output}; playing strength is not yet measured")


class NodeLimit(Exception):
    pass


def terminal(board, ply):
    if board.is_checkmate():
        return -MATE + ply
    # Automatic draws only. A claimable draw remains playable in the CLI.
    if board.is_game_over(claim_draw=False):
        return 0
    return None


def move_order(board, move):
    victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    return (10 * PIECE_VALUES[victim or 0] - PIECE_VALUES[attacker] if board.is_capture(move) else 0) + PIECE_VALUES[move.promotion or 0]


def choose_move(board, model, depth=3, max_nodes=20000):
    """Iterative alpha-beta with bounded quiescence. Never calls Stockfish."""
    if depth < 1 or max_nodes < 1:
        raise ValueError("Depth and node budget must be positive")
    if terminal(board, 0) is not None:
        return None, 0
    nodes = 0

    def search(remaining, alpha, beta, ply, quiet=4):
        nonlocal nodes
        if nodes >= max_nodes:
            raise NodeLimit
        nodes += 1
        result = terminal(board, ply)
        if result is not None:
            return result
        in_check = board.is_check()
        if remaining <= 0:
            score = float(model.predict(features(board))) * 1000 * (1 if board.turn else -1)
            # ponytail: bounded quiescence can miss long tactics; extend with a larger search budget.
            if quiet <= 0:
                return score
            if not in_check:
                if score >= beta:
                    return score
                alpha = max(alpha, score)
        moves = sorted(board.legal_moves, key=lambda m: move_order(board, m), reverse=True)
        if remaining <= 0 and not in_check:
            moves = [m for m in moves if board.is_capture(m) or m.promotion]
        for move in moves:
            board.push(move)
            try:
                score = -search(remaining - 1, -beta, -alpha, ply + 1,
                                quiet - 1 if remaining <= 0 else quiet)
            finally:
                board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    moves = sorted(board.legal_moves, key=lambda m: move_order(board, m), reverse=True)
    best = moves[0]
    for iteration in range(1, depth + 1):
        alpha, candidate = -MATE * 2, best
        moves.sort(key=lambda m: m == best, reverse=True)
        try:
            for move in moves:
                board.push(move)
                try:
                    score = -search(iteration - 1, -MATE * 2, -alpha, 1)
                finally:
                    board.pop()
                if score > alpha:
                    alpha, candidate = score, move
        except NodeLimit:
            break
        best = candidate
    return best, nodes


def play(args):
    board = board_from_fen(args.fen)
    move, nodes = choose_move(board, Model.load(args.model), args.depth, args.max_nodes)
    print(json.dumps({"move": move.uci() if move else None, "nodes": nodes}))


def benchmark(args):
    """Compare chosen moves to Stockfish on a separate, unseen dataset."""
    model = Model.load(args.model)
    rows = load_rows(args.data)
    random.Random(args.seed).shuffle(rows)
    losses, matches = [], 0
    with chess.engine.SimpleEngine.popen_uci(str(args.engine)) as engine:
        engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        for row in rows:
            board = board_from_fen(row["fen"])
            move, _ = choose_move(board, model, args.depth, args.max_nodes)
            if move is None:
                continue
            limit = chess.engine.Limit(nodes=args.nodes)
            reference = engine.analyse(board, limit)
            forced = engine.analyse(board, limit, root_moves=[move])
            best_cp = reference["score"].pov(board.turn).score(mate_score=10000)
            move_cp = forced["score"].pov(board.turn).score(mate_score=10000)
            losses.append(max(0, best_cp - move_cp))
            matches += move == reference["pv"][0]
            if len(losses) >= args.positions:
                break
    if not losses:
        raise ValueError("No playable benchmark positions")
    print(json.dumps({"positions": len(losses), "mean_cp_loss": float(np.mean(losses)),
                      "best_move_agreement": matches / len(losses)}))


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Run Stockfish to create training data")
    gen.add_argument("--output", default="positions.jsonl")
    gen.add_argument("--positions", type=positive, default=10000)
    fit = commands.add_parser("train", help="Train on an existing labeled dataset")
    fit.add_argument("--data", default="positions.jsonl")
    fit.add_argument("--output", default="model.npz")
    fit.add_argument("--epochs", type=positive, default=20)
    fit.add_argument("--batch-size", type=positive, default=256)
    fit.add_argument("--learning-rate", type=float, default=0.001)
    move = commands.add_parser("play", help="Print one legal move, without Stockfish")
    move.add_argument("--fen", default=chess.STARTING_FEN)
    bench = commands.add_parser("benchmark", help="Measure move quality against Stockfish")
    bench.add_argument("--data", required=True)
    bench.add_argument("--positions", type=positive, default=100)
    for sub in (gen, bench):
        sub.add_argument("--engine", type=Path, default=ENGINE)
        sub.add_argument("--nodes", type=positive, default=10000)
        sub.add_argument("--threads", type=positive, default=1)
        sub.add_argument("--hash-mb", type=positive, default=64)
    for sub in (gen, fit, bench):
        sub.add_argument("--seed", type=int, default=42)
    for sub in (move, bench):
        sub.add_argument("--model", default="model.npz")
        sub.add_argument("--depth", type=positive, default=3)
        sub.add_argument("--max-nodes", type=positive, default=20000)
    args = parser.parse_args()
    if args.command == "train" and (not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Learning rate must be finite and positive")
    try:
        {"generate": generate, "train": train, "play": play, "benchmark": benchmark}[args.command](args)
    except (ValueError, OSError, chess.engine.EngineError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
