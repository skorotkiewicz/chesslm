// Run with: node --test test_chess_web.js
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {test} = require('node:test');
const vm = require('node:vm');

const source = readFileSync(`${__dirname}/web/chess_web.html`, 'utf8')
  .match(/<script>\s*([\s\S]*?)<\/script>/)[1]
  .replace(/\nnewGame\(\);\s*$/, '');

function page() {
  const elements = new Map();
  const element = () => ({
    dataset: {}, classList: {toggle() {}}, value: 'white',
    contains() { return false; }, replaceChildren() {}, setAttribute() {},
    append() {}, addEventListener() {},
  });
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    querySelector() { return {content: 'server'}; },
    createElement: element,
  };
  const context = vm.createContext({document});
  const run = code => vm.runInContext(code, context);
  run(source);
  run(`state = {over: true, moves: [], pieces: {}, legal: [], reason: 'checkmate'};`);
  return {run, get: id => document.getElementById(id)};
}

for (const side of ['white', 'black']) {
  test(`watch winners ignore the selected human side: ${side}`, () => {
    const {run, get} = page();
    run(`human = '${side}'; request = () => new Promise(() => {}); $('watch').onclick();`);
    // Keep this check synchronous and independent of HTTP.
    for (const [result, heading] of [['1-0', 'Model wins'], ['0-1', 'Stockfish wins'], ['1/2-1/2', 'Draw']]) {
      run(`state = {over: true, result: '${result}', moves: [], pieces: {}, legal: [], reason: 'checkmate'}; render();`);
      assert.equal(get('heading').textContent, heading);
    }
  });

  test(`human game winner labels stay unchanged: ${side}`, () => {
    const {run, get} = page();
    run(`human = '${side}';`);
    for (const result of ['1-0', '0-1', '1/2-1/2']) {
      run(`state.result = '${result}'; render();`);
      const expected = result === '1/2-1/2' ? 'Draw'
        : (result === '1-0') === (side === 'white') ? 'You win' : 'Model wins';
      assert.equal(get('heading').textContent, expected);
    }
  });
}

test('watch starts fresh, cannot restart during a request, and keeps winner identity after stop', async () => {
  const {run, get} = page();
  run(`
    state = {over: false, moves: ['e2e4'], pieces: {}, legal: [], turn: 'black'};
    lastNodes = 99;
    var requests = [], finish;
    request = (moves, think, watch) => {
      requests.push({moves, think, watch});
      return new Promise(resolve => { finish = resolve; });
    };
    $('watch').onclick();
  `);
  assert.equal(run('JSON.stringify(requests)'), '[{"moves":[],"think":false,"watch":true}]');
  assert.equal(run('state'), null);
  assert.equal(run('lastNodes'), null);
  run(`$('watch').onclick();`); // Stop while Stockfish is still answering.
  assert.equal(run('watching'), false);
  run(`$('watch').onclick();`); // Busy: must not restart the loop.
  assert.equal(run('watching'), false);
  run(`finish({over: true, result: '0-1', reason: 'checkmate', moves: [], pieces: {}, legal: [], nodes: 100});`);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(get('heading').textContent, 'Stockfish wins');
  assert.equal(run('requests.length'), 1);

  run(`
    requests = [];
    request = async (moves, think, watch) => {
      requests.push({moves, think, watch});
      return {over: false, moves: [], pieces: {}, legal: [], turn: 'white'};
    };
    newGame();
  `);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run('JSON.stringify(requests)'), '[{"moves":[],"think":false}]');
  run(`state.over = true; state.result = '0-1'; render();`);
  assert.equal(get('heading').textContent, 'Model wins');
});
