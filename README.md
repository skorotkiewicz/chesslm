# Tiny chess model

A 100,353-parameter evaluator with alpha-beta search. Its float32 weights occupy
401,412 bytes, plus a small NPZ header. It uses NumPy and python-chess, not a
language model or GPU framework.

**No training or Stockfish data generation has been run on this machine.**
There is no trained checkpoint yet. Tests use hand-set zero weights. Passing
those tests says nothing about learned playing strength or Elo.

## Setup on the training machine

Use Python 3.10 or newer. Copy this project and the supplied Stockfish directory.
Keep Stockfish's license files with its binary when redistributing it.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The default teacher is `stockfish/stockfish-linux-x86-64-universal`, resolved
relative to `chesslm.py`. It requires x86-64 Linux. Use `--engine /path/to/stockfish`
for another location or a platform-compatible binary.

## Generate and train elsewhere

These commands are examples to run manually on the training machine. Nothing
starts automatically. Output files must not already exist.

```sh
python chesslm.py generate --positions 100000 --nodes 20000 --threads 2 --output positions.jsonl
OPENBLAS_NUM_THREADS=2 python chesslm.py train --data positions.jsonl --epochs 30 --output model.npz
```

Stockfish labels self-play positions with White's centipawn evaluation. Each
position uses a three-line search. Opening choices vary among those lines, with
occasional variation later. Search nodes bound teacher work, not elapsed time.
Generation time depends on the training machine. No time estimate is measured.

The model takes 12 piece planes plus turn, castling rights, legal en passant
file, and halfmove clock. A 128-unit tanh hidden layer learns a correction to a
fixed material evaluator. The output is `tanh(centipawns / 600)`. Mate labels use
+/-10,000 centipawns. Training minimizes squared error with Adam and saves the
lowest validation-loss weights, including the initial weights if no epoch helps.

Validation holds out entire games and removes positions shared with those games
from training. At least two games and some distinct positions are required.
The dataset is loaded into memory, roughly 313 MB of input arrays per 100,000
positions plus JSON rows and temporary arrays. Larger corpora need more RAM.

## Play without Stockfish

Copy the resulting `model.npz` back to the playing machine, then run:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python chesslm.py play --model model.npz --depth 3 --max-nodes 20000
```

Add `--fen '...'` to choose a move in another position. The command returns JSON
with a UCI move and visited node count. It returns `null` for an ended game.
This is a one-move CLI, not a UCI engine server. No Stockfish process runs during
play. Model weights fit under 0.5 MB; Python, NumPy, and process memory do not.

Search uses iterative deepening, capture ordering, and four quiescence plies.
A node limit can leave only a shallow completed iteration, or a legal fallback
if no iteration completes. Long tactics remain a limitation. Automatic draws
are recognized; claiming optional draws is not implemented. A FEN cannot carry
prior repetition history.

## Measure strength on the training machine

Generate a separate corpus with a different seed. Do not train on this file.

```sh
python chesslm.py generate --seed 9001 --positions 5000 --nodes 20000 --output benchmark.jsonl
OPENBLAS_NUM_THREADS=1 python chesslm.py benchmark --model model.npz --data benchmark.jsonl --positions 100 --nodes 50000
```

The report gives best-move agreement and mean centipawn loss against Stockfish.
These are noisy, node-limited estimates, not Elo. Different seeds can still
produce repeated openings. Match testing is needed before making strength
claims. A tiny distilled model should not be expected to match Stockfish.

## Local checks, no training

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest -v
.venv/bin/python chesslm.py --help
```

Checks cover encoding, checkpoint size and loading, invalid positions, held-out
game separation, node limits, promotions, check evasion, forced mates, and a free
queen capture. They do not invoke data generation, the optimizer, or Stockfish.
The generation and training commands still require end-to-end verification on
the training machine.
