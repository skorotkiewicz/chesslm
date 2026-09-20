"""Shared move validation and board responses for HTTP and browser play."""
import chess


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
