# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0,<3", "python-chess==1.999"]
# ///
"""Play against model.npz. Inference only; no Stockfish or training."""
import argparse
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import ttk
from zipfile import BadZipFile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import chess
from chesslm import Model, choose_move, positive

CELL, MARGIN = 64, 24
BOARD_SIZE = CELL * 8 + MARGIN * 2
LIGHT, DARK = "#eee7d5", "#759078"
INK = "#202721"


class ChessGame:
    def __init__(self, root, model, human=chess.WHITE, depth=3, max_nodes=20000, quiescence_depth=4):
        self.root, self.model = root, model
        self.human, self.depth, self.max_nodes = human, depth, max_nodes
        self.quiescence_depth = quiescence_depth
        self.results = Queue()
        self.thinking = False
        self.cursor = None
        root.title("Tiny chess")
        root.resizable(False, False)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=12, font=("DejaVu Sans", 12)).pack()
        self.canvas = tk.Canvas(root, width=BOARD_SIZE, height=BOARD_SIZE,
                                background=INK, highlightthickness=0, takefocus=True)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        for key in ("Left", "Right", "Up", "Down", "Return", "space"):
            self.canvas.bind(f"<{key}>", self.on_key)
        self.promotions = ttk.Frame(root)
        for label, piece in (("Queen", chess.QUEEN), ("Rook", chess.ROOK),
                             ("Bishop", chess.BISHOP), ("Knight", chess.KNIGHT)):
            ttk.Button(self.promotions, text=label,
                       command=lambda p=piece: self.promote(p)).pack(side="left", padx=3)
        controls = ttk.Frame(root, padding=10)
        controls.pack()
        self.restart = ttk.Button(controls, text="New game", command=self.reset)
        self.restart.pack(side="left", padx=6)
        ttk.Button(controls, text="Quit", command=root.destroy).pack(side="left", padx=6)
        ttk.Label(root, text="Click piece, then destination. Or use arrows and Enter.",
                  padding=(10, 0, 10, 10)).pack()
        self.reset()
        root.after(50, self.poll)

    def reset(self):
        # Keep one worker at a time; reset is disabled until its result arrives.
        if self.thinking:
            return
        self.board = chess.Board()
        self.selected, self.pending, self.error = None, [], None
        self.promotions.pack_forget()
        self.refresh()

    def square_at(self, x, y):
        col, row = (x - MARGIN) // CELL, (y - MARGIN) // CELL
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        return chess.square(col, 7 - row) if self.human else chess.square(7 - col, row)

    def square_center(self, square):
        col, row = chess.square_file(square), chess.square_rank(square)
        col, row = (col, 7 - row) if self.human else (7 - col, row)
        return MARGIN + col * CELL + CELL // 2, MARGIN + row * CELL + CELL // 2

    def on_click(self, event):
        self.canvas.focus_set()
        self.cursor = None
        self.select(self.square_at(event.x, event.y))

    def on_key(self, event):
        if self.cursor is None:
            self.cursor = chess.E2 if self.human else chess.E7
        if event.keysym in ("Return", "space"):
            self.select(self.cursor)
        else:
            x, y = self.square_center(self.cursor)
            dx, dy = {"Left": (-CELL, 0), "Right": (CELL, 0),
                      "Up": (0, -CELL), "Down": (0, CELL)}[event.keysym]
            square = self.square_at(x + dx, y + dy)
            if square is not None:
                self.cursor = square
            self.draw()
        return "break"

    def select(self, square):
        if (square is None or self.thinking or self.error or self.pending
                or self.board.turn != self.human or self.board.is_game_over()):
            return
        if self.selected is not None:
            moves = [m for m in self.board.legal_moves
                     if m.from_square == self.selected and m.to_square == square]
            if moves:
                if moves[0].promotion:
                    self.pending = moves
                    self.promotions.pack(before=self.restart.master, pady=8)
                    self.status.set("Choose a promotion piece")
                    self.promotions.winfo_children()[0].focus_set()
                else:
                    self.push(moves[0])
                return
        piece = self.board.piece_at(square)
        self.selected = square if piece and piece.color == self.human and square != self.selected else None
        self.draw()

    def promote(self, piece):
        move = next((m for m in self.pending if m.promotion == piece), None)
        if move is not None:
            self.pending = []
            self.promotions.pack_forget()
            self.canvas.focus_set()
            self.push(move)

    def push(self, move):
        self.board.push(move)
        self.selected = None
        self.refresh()

    def refresh(self):
        outcome = self.board.outcome()
        if self.error:
            self.status.set(f"AI error: {self.error}. Start a new game to retry.")
        elif outcome:
            winner = "Draw" if outcome.winner is None else ("You win" if outcome.winner == self.human else "Model wins")
            reason = outcome.termination.name.replace("_", " ").lower()
            self.status.set(f"{winner}: {reason}")
        elif self.board.turn == self.human:
            color = "White" if self.human else "Black"
            self.status.set(f"Your turn ({color})" + (". Check!" if self.board.is_check() else ""))
        else:
            self.status.set("Model is thinking...")
            if not self.thinking:
                self.thinking = True
                Thread(target=self.search, args=(self.board.copy(),), daemon=True).start()
        self.restart.configure(state="disabled" if self.thinking else "normal")
        self.draw()

    def search(self, board):
        # Workers only use a board copy and queue, never Tk objects.
        try:
            move, _ = choose_move(board, self.model, self.depth, self.max_nodes,
                                  quiescence_depth=self.quiescence_depth)
            self.results.put((move, None))
        except Exception as error:
            self.results.put((None, str(error)))

    def poll(self):
        try:
            move, error = self.results.get_nowait()
        except Empty:
            pass
        else:
            self.thinking = False
            if error is not None or move not in self.board.legal_moves:
                self.error = error or "search returned no legal move"
                self.refresh()
            else:
                self.push(move)
        self.root.after(50, self.poll)

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        last = self.board.peek() if self.board.move_stack else None
        targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}
        for square in chess.SQUARES:
            x, y = self.square_center(square)
            color = LIGHT if (chess.square_file(square) + chess.square_rank(square)) % 2 else DARK
            if last and square in (last.from_square, last.to_square):
                color = "#c8bd78"
            if square == self.selected:
                color = "#e6b85a"
            if self.board.is_check() and square == self.board.king(self.board.turn):
                color = "#cf7567"
            canvas.create_rectangle(x - CELL // 2, y - CELL // 2, x + CELL // 2,
                                    y + CELL // 2, fill=color, outline="")
            piece = self.board.piece_at(square)
            if piece:
                glyph = chess.Piece(piece.piece_type, chess.BLACK).unicode_symbol()
                fill, edge = ("#fffaf0", INK) if piece.color else (INK, "#fffaf0")
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    canvas.create_text(x + dx, y + dy, text=glyph, fill=edge, font=("DejaVu Sans", 40))
                canvas.create_text(x, y, text=glyph, fill=fill, font=("DejaVu Sans", 40))
            if square in targets:
                radius = 27 if piece else 7
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                   outline=INK, width=3, fill="" if piece else INK)
            if square == self.cursor:
                canvas.create_rectangle(x - 29, y - 29, x + 29, y + 29, outline=INK, width=3, dash=(4, 3))
        for i in range(8):
            square = self.square_at(MARGIN + i * CELL + 1, MARGIN + 1)
            canvas.create_text(MARGIN + i * CELL + CELL // 2, BOARD_SIZE - 12,
                               text=chess.FILE_NAMES[chess.square_file(square)], fill=LIGHT)
            square = self.square_at(MARGIN + 1, MARGIN + i * CELL + 1)
            canvas.create_text(12, MARGIN + i * CELL + CELL // 2,
                               text=chess.square_rank(square) + 1, fill=LIGHT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument("--color", choices=("white", "black"), default="white")
    parser.add_argument("--depth", type=positive, default=3)
    parser.add_argument("--max-nodes", type=positive, default=20000)
    parser.add_argument("--quiescence-depth", type=positive, default=4,
                        help="Maximum extra tactical plies, within --max-nodes (default: 4)")
    args = parser.parse_args()
    try:
        model = Model.load(args.model)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        parser.exit(1, f"Cannot load {args.model}: {error}\n")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(1, f"Cannot open a desktop window: {error}\n")
    ChessGame(root, model, human=args.color == "white", depth=args.depth, max_nodes=args.max_nodes,
              quiescence_depth=args.quiescence_depth)
    root.mainloop()


if __name__ == "__main__":
    main()
