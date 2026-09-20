// The model and search run in this worker, not on a server or the UI thread.
import {loadPyodide} from 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs';

try {
  const pyodide = await loadPyodide({indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/'});
  await pyodide.loadPackage('numpy');
  for (const name of ['chess.zip', 'chesslm.py', 'chess_position.py', 'model.npz']) {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Cannot load ${name}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (name === 'chess.zip') pyodide.unpackArchive(bytes, 'zip');
    else pyodide.FS.writeFile(name, bytes);
  }
  pyodide.runPython(`
import json
from chesslm import Model, choose_move
from chess_position import replay, position
model = Model.load('model.npz')

def browser_request(payload):
    data = json.loads(payload)
    board = replay(data['moves'])
    nodes = 0
    if data['think'] and not board.is_game_over():
        # ponytail: fixed browser budget; expose controls if users need deeper search.
        move, nodes = choose_move(board, model, depth=2, max_nodes=2000)
        if move not in board.legal_moves:
            raise ValueError('Search returned an illegal move')
        board.push(move)
    return json.dumps(position(board, nodes))
`);
  const request = pyodide.globals.get('browser_request');
  self.onmessage = ({data: {id, moves, think}}) => {
    try {
      self.postMessage({id, result: JSON.parse(request(JSON.stringify({moves, think})))});
    } catch (error) {
      self.postMessage({id, error: error.message});
    }
  };
  self.postMessage({ready: true});
} catch (error) {
  self.postMessage({fatal: `Could not load the browser model. Check your connection and reload. ${error.message}`});
}
