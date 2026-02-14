// ai_worker.js
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js");

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.simd = false;

const BOARD_SIZE = 5;
const ACTION_COUNT = 200;
const PLANE = BOARD_SIZE * BOARD_SIZE;
const RESTRICTED_XY = new Set([15, 20, 21, 3, 4, 9]); // (0,3)(0,4)(1,4)(3,0)(4,0)(4,1)
const LINE_WEIGHTS = [0, 1, 4, 16, 64, 5000];

let neuralSession = null;
let useNeural = false;
let modelInputName = "input";
let modelOutputName = "output";

const SHAPES = [
  [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }],
  [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }, { x: 0, y: -1, z: 0 }],
  [{ x: 0, y: 0, z: 0 }, { x: -1, y: 0, z: 0 }, { x: 0, y: -1, z: 0 }],
  [{ x: 0, y: 0, z: 0 }, { x: -1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }],
  [{ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }, { x: 1, y: 0, z: 1 }],
  [{ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }, { x: -1, y: 0, z: 1 }],
  [{ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }, { x: 0, y: 1, z: 1 }],
  [{ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }, { x: 0, y: -1, z: 1 }],
];

function idx2(x, y) {
  return y * BOARD_SIZE + x;
}
function idx3(x, y, z) {
  return z * PLANE + y * BOARD_SIZE + x;
}
function buildLines() {
  const lines = [];
  for (let y = 0; y < BOARD_SIZE; y++) {
    lines.push([idx2(0, y), idx2(1, y), idx2(2, y), idx2(3, y), idx2(4, y)]);
  }
  for (let x = 0; x < BOARD_SIZE; x++) {
    lines.push([idx2(x, 0), idx2(x, 1), idx2(x, 2), idx2(x, 3), idx2(x, 4)]);
  }
  lines.push([idx2(0, 0), idx2(1, 1), idx2(2, 2), idx2(3, 3), idx2(4, 4)]);
  lines.push([idx2(4, 0), idx2(3, 1), idx2(2, 2), idx2(1, 3), idx2(0, 4)]);
  return lines;
}
const LINES = buildLines();

function cloneCells(cells) {
  return cells.map((c) => ({ x: c.x, y: c.y, z: c.z }));
}
function cloneBoard(board) {
  const out = new Array(BOARD_SIZE);
  for (let z = 0; z < BOARD_SIZE; z++) {
    out[z] = new Array(BOARD_SIZE);
    for (let y = 0; y < BOARD_SIZE; y++) out[z][y] = board[z][y].slice();
  }
  return out;
}
function cloneBlocks(blocks) {
  return blocks.map((b) => ({
    id: b.id,
    player: b.player,
    isFixed: !!b.isFixed,
    shapeIdx: b.shapeIdx ?? 0,
    cells: cloneCells(b.cells || []),
  }));
}
function cloneState(state) {
  return {
    board: cloneBoard(state.board),
    blocks: cloneBlocks(state.blocks),
    blocksLeft: { 1: state.blocksLeft[1], 2: state.blocksLeft[2] },
    phase: state.phase,
    player: state.player,
    turnCount: state.turnCount,
    winner: state.winner,
    nextId: state.nextId,
  };
}

function checkValidity(board, cells, turnCount, ignoreSet = null) {
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    if (c.x < 0 || c.x > 4 || c.y < 0 || c.y > 4 || c.z < 0 || c.z > 4) return false;
    const id = idx3(c.x, c.y, c.z);
    if (board[c.z][c.y][c.x] !== 0 && !(ignoreSet && ignoreSet.has(id))) return false;
  }

  let ground = 0;
  for (let i = 0; i < cells.length; i++) if (cells[i].z === 0) ground++;
  if (ground !== 1 && ground !== 3) return false;

  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    if (c.z > 0) {
      const belowId = idx3(c.x, c.y, c.z - 1);
      const hasSupport =
        board[c.z - 1][c.y][c.x] !== 0 && !(ignoreSet && ignoreSet.has(belowId));

      let isSelf = false;
      for (let j = 0; j < cells.length; j++) {
        if (cells[j].x === c.x && cells[j].y === c.y && cells[j].z === c.z - 1) {
          isSelf = true;
          break;
        }
      }
      if (!hasSupport && !isSelf) return false;
    }
  }

  if (!ignoreSet && turnCount < 2) {
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      if (c.z === 0 && RESTRICTED_XY.has(idx2(c.x, c.y))) return false;
    }
  }

  return true;
}

function getCells(px, py, shapeIdx) {
  const s = SHAPES[shapeIdx];
  return s.map((d) => ({ x: px + d.x, y: py + d.y, z: d.z }));
}

function getLandingCells(board, actionIdx, turnCount, ignoreSet = null) {
  const shapeIdx = actionIdx % 8;
  const px = Math.floor(actionIdx / 8) % 5;
  const py = Math.floor(actionIdx / 40);
  const base = getCells(px, py, shapeIdx);

  for (let dz = 0; dz < 5; dz++) {
    const test = base.map((c) => ({ x: c.x, y: c.y, z: c.z + dz }));
    let out = false;
    for (let i = 0; i < test.length; i++) {
      if (test[i].z > 4) {
        out = true;
        break;
      }
    }
    if (out) break;
    if (checkValidity(board, test, turnCount, ignoreSet)) {
      return { cells: test, shapeIdx, actionIdx };
    }
  }
  return null;
}

function canPickBlock(board, block) {
  if (block.isFixed) return false;
  for (let i = 0; i < block.cells.length; i++) {
    const c = block.cells[i];
    if (c.z >= 4) continue;
    const above = board[c.z + 1][c.y][c.x];
    if (above !== 0) {
      let isSelf = false;
      for (let j = 0; j < block.cells.length; j++) {
        const s = block.cells[j];
        if (s.x === c.x && s.y === c.y && s.z === c.z + 1) {
          isSelf = true;
          break;
        }
      }
      if (!isSelf) return false;
    }
  }
  return true;
}

function sameCells(a, b) {
  if (a.length !== b.length) return false;
  const setA = new Set(a.map((c) => idx3(c.x, c.y, c.z)));
  for (let i = 0; i < b.length; i++) {
    if (!setA.has(idx3(b[i].x, b[i].y, b[i].z))) return false;
  }
  return true;
}

function buildTopMap(board) {
  const top = new Int8Array(25);
  for (let y = 0; y < 5; y++) {
    for (let x = 0; x < 5; x++) {
      for (let z = 4; z >= 0; z--) {
        const v = board[z][y][x];
        if (v !== 0) {
          top[idx2(x, y)] = v;
          break;
        }
      }
    }
  }
  return top;
}

function checkWin(board) {
  const top = buildTopMap(board);
  for (let i = 0; i < LINES.length; i++) {
    const line = LINES[i];
    const p = top[line[0]];
    if (
      p !== 0 &&
      top[line[1]] === p &&
      top[line[2]] === p &&
      top[line[3]] === p &&
      top[line[4]] === p
    ) {
      return p;
    }
  }
  return 0;
}

function linePotential(top, player) {
  const opp = player === 1 ? 2 : 1;
  let score = 0;
  for (let i = 0; i < LINES.length; i++) {
    const line = LINES[i];
    let mine = 0;
    let other = 0;
    for (let k = 0; k < 5; k++) {
      const v = top[line[k]];
      if (v === player) mine++;
      else if (v === opp) other++;
    }
    if (other === 0) score += LINE_WEIGHTS[mine];
  }
  return score;
}

function evaluateStatic(board, rootPlayer) {
  const winner = checkWin(board);
  if (winner === rootPlayer) return 1.0;
  if (winner !== 0) return -1.0;

  const top = buildTopMap(board);
  const opp = rootPlayer === 1 ? 2 : 1;

  let score = linePotential(top, rootPlayer) - linePotential(top, opp);
  const center = [12, 7, 11, 13, 17];
  for (let i = 0; i < center.length; i++) {
    const v = top[center[i]];
    if (v === rootPlayer) score += 1.5;
    else if (v === opp) score -= 1.5;
  }

  let val = score / 100;
  if (val > 0.95) val = 0.95;
  if (val < -0.95) val = -0.95;
  return val;
}

function moveKey(move) {
  if (move.type === "place") return `p:${move.actionIdx}`;
  return `m:${move.fromId}:${move.actionIdx}`;
}

function boardApplyMove(board, move, player) {
  const backup = new Map();

  function writeCell(c, value) {
    const id = idx3(c.x, c.y, c.z);
    if (!backup.has(id)) backup.set(id, board[c.z][c.y][c.x]);
    board[c.z][c.y][c.x] = value;
  }

  if (move.type === "move") {
    for (let i = 0; i < move.originCells.length; i++) writeCell(move.originCells[i], 0);
  }
  for (let i = 0; i < move.cells.length; i++) writeCell(move.cells[i], player);

  return backup;
}

function boardUndo(board, backup) {
  for (const [id, val] of backup.entries()) {
    const z = Math.floor(id / PLANE);
    const rem = id - z * PLANE;
    const y = Math.floor(rem / BOARD_SIZE);
    const x = rem - y * BOARD_SIZE;
    board[z][y][x] = val;
  }
}

function hasImmediatePlacementWin(board, player, turnCount) {
  for (let i = 0; i < ACTION_COUNT; i++) {
    const res = getLandingCells(board, i, turnCount, null);
    if (!res) continue;
    const move = { type: "place", cells: res.cells, shapeIdx: res.shapeIdx, actionIdx: i };
    const backup = boardApplyMove(board, move, player);
    const win = checkWin(board) === player;
    boardUndo(board, backup);
    if (win) return true;
  }
  return false;
}

function findImmediateWinningMove(board, candidates, player) {
  for (let i = 0; i < candidates.length; i++) {
    const move = candidates[i];
    const backup = boardApplyMove(board, move, player);
    const win = checkWin(board) === player;
    boardUndo(board, backup);
    if (win) return move;
  }
  return null;
}

function findBlockingPlacementMove(board, candidates, player, turnCount) {
  const opp = player === 1 ? 2 : 1;
  if (!hasImmediatePlacementWin(board, opp, turnCount)) return null;

  for (let i = 0; i < candidates.length; i++) {
    const move = candidates[i];
    if (move.type !== "place") continue;
    const backup = boardApplyMove(board, move, player);
    const oppStillWins = hasImmediatePlacementWin(board, opp, turnCount + 1);
    boardUndo(board, backup);
    if (!oppStillWins) return move;
  }
  return null;
}

function generateCandidates(state) {
  const board = state.board;
  const blocks = state.blocks;
  const player = state.player;
  const phase = state.phase;
  const blocksLeft = state.blocksLeft;
  const turnCount = state.turnCount;

  const candidates = [];

  if (phase === "PLACEMENT") {
    if ((blocksLeft[player] || 0) <= 0) return candidates;
    for (let i = 0; i < ACTION_COUNT; i++) {
      const res = getLandingCells(board, i, turnCount, null);
      if (res) {
        candidates.push({
          type: "place",
          cells: res.cells,
          shapeIdx: res.shapeIdx,
          actionIdx: i,
        });
      }
    }
    return candidates;
  }

  for (let bi = 0; bi < blocks.length; bi++) {
    const b = blocks[bi];
    if (b.player !== player || b.isFixed) continue;
    if (!canPickBlock(board, b)) continue;

    const ignoreSet = new Set(b.cells.map((c) => idx3(c.x, c.y, c.z)));
    for (let i = 0; i < ACTION_COUNT; i++) {
      const res = getLandingCells(board, i, turnCount, ignoreSet);
      if (!res) continue;
      if (sameCells(res.cells, b.cells)) continue;

      candidates.push({
        type: "move",
        fromId: b.id,
        originCells: cloneCells(b.cells),
        cells: res.cells,
        shapeIdx: res.shapeIdx,
        actionIdx: i,
      });
    }
  }

  return candidates;
}

function applyMoveToState(state, move) {
  const player = state.player;

  if (move.type === "place") {
    for (let i = 0; i < move.cells.length; i++) {
      const c = move.cells[i];
      state.board[c.z][c.y][c.x] = player;
    }
    state.blocks.push({
      id: state.nextId--,
      player,
      isFixed: false,
      shapeIdx: move.shapeIdx,
      cells: cloneCells(move.cells),
    });
    state.blocksLeft[player] = Math.max(0, (state.blocksLeft[player] || 0) - 1);
  } else {
    const bi = state.blocks.findIndex((b) => b.id === move.fromId && b.player === player);
    if (bi < 0) return false;
    const block = state.blocks[bi];
    if (block.isFixed) return false;

    for (let i = 0; i < block.cells.length; i++) {
      const c = block.cells[i];
      state.board[c.z][c.y][c.x] = 0;
    }
    block.cells = cloneCells(move.cells);
    block.shapeIdx = move.shapeIdx;
    for (let i = 0; i < block.cells.length; i++) {
      const c = block.cells[i];
      state.board[c.z][c.y][c.x] = player;
    }
  }

  const winner = checkWin(state.board);
  state.winner = winner;
  if (winner !== 0) return true;

  state.turnCount += 1;
  state.player = player === 1 ? 2 : 1;

  if (state.phase === "PLACEMENT") {
    if (state.blocksLeft[1] === 0 && state.blocksLeft[2] === 0) state.phase = "MOVEMENT";
    else if (state.blocksLeft[state.player] === 0) state.phase = "MOVEMENT";
  }

  return true;
}

class TreeNode {
  constructor(move = null, prior = 1) {
    this.move = move;
    this.prior = prior;
    this.children = [];
    this.unexpanded = null;
    this.N = 0;
    this.W = 0;
    this.Q = 0;
  }
}

function quickPrior(move) {
  if (move.type === "place") {
    let s = 1;
    for (let i = 0; i < move.cells.length; i++) {
      const c = move.cells[i];
      s += 5 - Math.abs(c.x - 2) - Math.abs(c.y - 2);
      if (c.z > 0) s += 0.6;
    }
    return Math.max(0.1, s);
  }
  return 1.0;
}

function makeMoveEntries(candidates, priorMap, maxWidth) {
  let entries = candidates.map((move) => ({
    move,
    prior: priorMap ? (priorMap.get(moveKey(move)) ?? quickPrior(move)) : quickPrior(move),
  }));

  entries.sort((a, b) => b.prior - a.prior);
  if (maxWidth > 0 && entries.length > maxWidth) entries = entries.slice(0, maxWidth);

  let sum = 0;
  for (let i = 0; i < entries.length; i++) sum += Math.max(entries[i].prior, 1e-6);
  if (sum <= 0) sum = entries.length;
  for (let i = 0; i < entries.length; i++) entries[i].prior = Math.max(entries[i].prior, 1e-6) / sum;

  return entries;
}

function pickEntryIndexByPrior(entries) {
  let r = Math.random();
  let acc = 0;
  for (let i = 0; i < entries.length; i++) {
    acc += entries[i].prior;
    if (r <= acc) return i;
  }
  return entries.length - 1;
}

function evaluateLeaf(state, rootPlayer) {
  const winner = state.winner || checkWin(state.board);
  if (winner === rootPlayer) return 1.0;
  if (winner !== 0) return -1.0;
  return evaluateStatic(state.board, rootPlayer);
}

function runPUCT(rootState, rootCandidates, rootPriorMap, opts) {
  const root = new TreeNode(null, 1);
  root.unexpanded = makeMoveEntries(rootCandidates, rootPriorMap, opts.rootWidth);
  if (root.unexpanded.length === 0) return { move: null, sims: 0 };

  const rootPlayer = rootState.player;
  const start = performance.now();
  let sims = 0;

  while (sims < opts.maxSims && performance.now() - start < opts.timeMs) {
    sims++;
    const sim = cloneState(rootState);
    const path = [root];
    let node = root;
    let depth = 0;

    while (depth < opts.maxDepth) {
      const winner = sim.winner || checkWin(sim.board);
      if (winner !== 0) {
        sim.winner = winner;
        break;
      }

      if (node.unexpanded === null) {
        const cands = generateCandidates(sim);
        node.unexpanded = makeMoveEntries(cands, null, opts.innerWidth);
      }

      if (node.unexpanded.length > 0) {
        const pickIdx = pickEntryIndexByPrior(node.unexpanded);
        const picked = node.unexpanded.splice(pickIdx, 1)[0];
        if (!applyMoveToState(sim, picked.move)) break;

        const child = new TreeNode(picked.move, picked.prior);
        node.children.push(child);
        node = child;
        path.push(node);
        depth++;
        break;
      }

      if (node.children.length === 0) break;

      const sqrtParent = Math.sqrt(node.N + 1);
      let bestChild = null;
      let bestScore = -Infinity;

      for (let i = 0; i < node.children.length; i++) {
        const child = node.children[i];
        const exploit = sim.player === rootPlayer ? child.Q : -child.Q;
        const explore = opts.cpuct * child.prior * sqrtParent / (1 + child.N);
        const score = exploit + explore;
        if (score > bestScore) {
          bestScore = score;
          bestChild = child;
        }
      }

      if (!bestChild) break;
      if (!applyMoveToState(sim, bestChild.move)) break;

      node = bestChild;
      path.push(node);
      depth++;
    }

    const value = evaluateLeaf(sim, rootPlayer);
    for (let i = 0; i < path.length; i++) {
      const n = path[i];
      n.N += 1;
      n.W += value;
      n.Q = n.W / n.N;
    }
  }

  let best = null;
  for (let i = 0; i < root.children.length; i++) {
    const c = root.children[i];
    if (!best || c.N > best.N || (c.N === best.N && c.Q > best.Q)) best = c;
  }

  return { move: best ? best.move : null, sims };
}

function buildInputTensor(board, player, phase) {
  const data = new Float32Array(1 * 3 * 5 * 5 * 5);
  const opp = player === 1 ? 2 : 1;
  const phaseVal = phase === "PLACEMENT" ? 1.0 : 0.0;

  let idx = 0;
  for (let z = 0; z < 5; z++) for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) data[idx++] = board[z][y][x] === player ? 1.0 : 0.0;
  for (let z = 0; z < 5; z++) for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) data[idx++] = board[z][y][x] === opp ? 1.0 : 0.0;
  for (let z = 0; z < 5; z++) for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) data[idx++] = phaseVal;

  return new ort.Tensor("float32", data, [1, 3, 5, 5, 5]);
}

async function inferPolicyLogits(board, player, phase) {
  if (!useNeural || !neuralSession) return null;

  const tensor = buildInputTensor(board, player, phase);
  const feeds = {};
  feeds[modelInputName] = tensor;

  const outputs = await neuralSession.run(feeds);
  let out = outputs[modelOutputName];
  if (!out) {
    const keys = Object.keys(outputs);
    if (keys.length === 0) return null;
    out = outputs[keys[0]];
  }
  return out && out.data ? out.data : null;
}

async function buildRootPriorMap(state, candidates) {
  if (!useNeural || state.phase !== "PLACEMENT" || candidates.length === 0) return null;

  try {
    const logits = await inferPolicyLogits(state.board, state.player, state.phase);
    if (!logits || logits.length < ACTION_COUNT) return null;

    let maxLogit = -Infinity;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      if (c.type !== "place") continue;
      if (logits[c.actionIdx] > maxLogit) maxLogit = logits[c.actionIdx];
    }
    if (maxLogit === -Infinity) return null;

    const map = new Map();
    let sum = 0;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      if (c.type !== "place") continue;
      const p = Math.exp(logits[c.actionIdx] - maxLogit);
      map.set(moveKey(c), p);
      sum += p;
    }
    if (sum <= 0) return null;
    for (const [k, v] of map.entries()) map.set(k, v / sum);

    return map;
  } catch (err) {
    return null;
  }
}

function pickBestByPrior(candidates, priorMap) {
  if (!priorMap) return null;
  let best = null;
  let maxP = -Infinity;
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const p = priorMap.get(moveKey(c));
    if (p != null && p > maxP) {
      maxP = p;
      best = c;
    }
  }
  return best;
}

function toRootState(gs) {
  return {
    board: cloneBoard(gs.board),
    blocks: cloneBlocks(gs.blocks || []),
    blocksLeft: {
      1: Number(gs.blocksLeft?.[1] ?? 0),
      2: Number(gs.blocksLeft?.[2] ?? 0),
    },
    phase: gs.phase === "MOVEMENT" ? "MOVEMENT" : "PLACEMENT",
    player: gs.player === 2 ? 2 : 1,
    turnCount: Number(gs.turnCount ?? 0),
    winner: 0,
    nextId: -1,
  };
}

self.onmessage = async function (e) {
  const msg = e.data || {};

  if (msg.type === "INIT") {
    try {
      neuralSession = await ort.InferenceSession.create(msg.url || "./omok_model.onnx");
      useNeural = true;
      if (neuralSession.inputNames && neuralSession.inputNames.length) modelInputName = neuralSession.inputNames[0];
      if (neuralSession.outputNames && neuralSession.outputNames.length) modelOutputName = neuralSession.outputNames[0];
      self.postMessage({ type: "INIT_OK" });
    } catch (err) {
      useNeural = false;
      neuralSession = null;
      self.postMessage({ type: "INIT_FAIL", error: err.toString() });
    }
    return;
  }

  if (msg.type !== "THINK") return;

  const reqId = typeof msg.reqId === "number" ? msg.reqId : null;
  const mode = msg.mode || "play";
  const gs = msg.gameState;

  if (!gs || !gs.board || !gs.blocksLeft || gs.player == null || !gs.phase) {
    self.postMessage({ type: "MOVE", reqId, mode, move: null, strategy: "bad_state" });
    return;
  }

  const state = toRootState(gs);
  state.winner = checkWin(state.board);

  let finalMove = null;
  let finalStrategy = "none";

  const candidates = generateCandidates(state);

  if (candidates.length > 0) {
    const winMove = findImmediateWinningMove(state.board, candidates, state.player);
    if (winMove) {
      finalMove = winMove;
      finalStrategy = "winning_move";
    } else if (state.phase === "PLACEMENT") {
      const blockMove = findBlockingPlacementMove(state.board, candidates, state.player, state.turnCount);
      if (blockMove) {
        finalMove = blockMove;
        finalStrategy = "blocking_move";
      }
    }

    if (!finalMove) {
      const priorMap = await buildRootPriorMap(state, candidates);

      const opts =
        state.phase === "PLACEMENT"
          ? {
              cpuct: 1.35,
              maxSims: mode === "hint" ? 220 : 320,
              timeMs: mode === "hint" ? 500 : 750,
              maxDepth: 8,
              rootWidth: 36,
              innerWidth: 16,
            }
          : {
              cpuct: 1.2,
              maxSims: mode === "hint" ? 140 : 200,
              timeMs: mode === "hint" ? 350 : 500,
              maxDepth: 6,
              rootWidth: 28,
              innerWidth: 12,
            };

      const mcts = runPUCT(state, candidates, priorMap, opts);
      if (mcts.move) {
        finalMove = mcts.move;
        finalStrategy = priorMap ? "puct_hybrid" : "mcts";
      } else {
        finalMove = pickBestByPrior(candidates, priorMap) || candidates[Math.floor(Math.random() * candidates.length)];
        finalStrategy = priorMap ? "neural_fallback" : "random_fallback";
      }
    }
  }

  self.postMessage({
    type: "MOVE",
    reqId,
    mode,
    move: finalMove,
    strategy: finalStrategy,
  });
};
