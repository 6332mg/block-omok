// 🧠 ai_worker.js

// 1. 호환성이 가장 좋은 1.14.0 버전으로 고정 (최신 버전은 보안 정책이 까다로움)
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js");

// 2. 부품(.wasm) 위치를 CDN으로 정확하게 지정
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";

// 🚨 [핵심 해결책] 멀티스레드 끄기
// Render 서버에는 보안 헤더(COOP/COEP)가 없으므로, 스레드를 1개로 제한해야만 작동함.
ort.env.wasm.numThreads = 1; 
ort.env.wasm.proxy = false; 

let neuralSession = null;
let useNeural = false;

// ... (이 아래 const SHAPES = ... 부터는 기존 코드 그대로 두세요) ...

const SHAPES = [
    [{x:0,y:0,z:0}, {x:1,y:0,z:0}, {x:0,y:1,z:0}], [{x:0,y:0,z:0}, {x:1,y:0,z:0}, {x:0,y:-1,z:0}],
    [{x:0,y:0,z:0}, {x:-1,y:0,z:0}, {x:0,y:-1,z:0}], [{x:0,y:0,z:0}, {x:-1,y:0,z:0}, {x:0,y:1,z:0}],
    [{x:0,y:0,z:0}, {x:0,y:0,z:1}, {x:1,y:0,z:1}], [{x:0,y:0,z:0}, {x:0,y:0,z:1}, {x:-1,y:0,z:1}],
    [{x:0,y:0,z:0}, {x:0,y:0,z:1}, {x:0,y:1,z:1}], [{x:0,y:0,z:0}, {x:0,y:0,z:1}, {x:0,y:-1,z:1}]
];

function checkValidity(board, player, cells, turnCount, ignoreCells=null) {
    const ignoreSet = new Set();
    if(ignoreCells) ignoreCells.forEach(c => ignoreSet.add(`${c.x},${c.y},${c.z}`));

    for(let c of cells) {
        if(c.x<0||c.x>4||c.y<0||c.y>4||c.z<0||c.z>4) return false;
        if(board[c.z][c.y][c.x] !== 0) {
            if(!ignoreSet.has(`${c.x},${c.y},${c.z}`)) return false;
        }
    }
    const ground = cells.filter(c=>c.z===0).length;
    if(ground!==1 && ground!==3) return false;

    for(let c of cells) {
        if(c.z > 0) {
            const hasSup = (board[c.z-1][c.y][c.x] !== 0) && (!ignoreSet.has(`${c.x},${c.y},${c.z-1}`));
            const isSelf = cells.some(sc => sc.x===c.x && sc.y===c.y && sc.z===c.z-1);
            if(!hasSup && !isSelf) return false;
        }
    }
    if(!ignoreCells && turnCount < 2) {
        const restricted = ["0,3","0,4","1,4","3,0","4,0","4,1"];
        if(cells.some(c=>c.z===0 && restricted.includes(`${c.x},${c.y}`))) return false;
    }
    return true;
}

function getCells(px, py, shIdx) {
    const s = SHAPES[shIdx];
    return s.map(d => ({x: px+d.x, y: py+d.y, z: d.z}));
}

function getLandingCells(board, player, actionIdx, turnCount, ignoreCells=null) {
    const sh = actionIdx % 8;
    const px = Math.floor(actionIdx / 8) % 5;
    const py = Math.floor(Math.floor(actionIdx / 8) / 5);
    const baseCells = getCells(px, py, sh);

    for(let dz=0; dz<5; dz++) {
        const testCells = baseCells.map(c => ({x: c.x, y: c.y, z: c.z + dz}));
        if(testCells.some(c => c.z > 4)) break;
        if(checkValidity(board, player, testCells, turnCount, ignoreCells)) {
            return { cells: testCells, shapeIdx: sh };
        }
    }
    return null;
}

// 🏆 승리 조건 체크 (Top View Simulation)
function checkWin(board) {
    const topMap = Array.from({length: 5}, () => Array(5).fill(0));
    for(let y=0; y<5; y++) {
        for(let x=0; x<5; x++) {
            for(let z=4; z>=0; z--) {
                if(board[z][y][x] !== 0) {
                    topMap[y][x] = board[z][y][x];
                    break;
                }
            }
        }
    }
    const dirs = [{dx:1, dy:0}, {dx:0, dy:1}, {dx:1, dy:1}, {dx:1, dy:-1}];
    for(let y=0; y<5; y++) {
        for(let x=0; x<5; x++) {
            let c = topMap[y][x];
            if(c === 0) continue;
            for(let d of dirs) {
                let cnt = 1;
                for(let k=1; k<5; k++) {
                    let nx = x + d.dx*k;
                    let ny = y + d.dy*k;
                    if(nx>=0 && nx<5 && ny>=0 && ny<5 && topMap[ny][nx] === c) cnt++;
                    else break;
                }
                if(cnt === 5) return c;
            }
        }
    }
    return 0;
}

// 🧠 1수 앞을 내다보는 시뮬레이션 (공격 & 방어 통합)
function findSmartMove(board, blocks, player, phase, blocksLeft, turnCount) {
    const opponent = player === 1 ? 2 : 1;
    let candidates = [];

    // [1] 후보 수집
    if (phase === 'PLACEMENT') {
        for(let i=0; i<200; i++) {
            const res = getLandingCells(board, player, i, turnCount);
            if(res) candidates.push({ type: 'place', cells: res.cells, shapeIdx: res.shapeIdx, actionIdx: i });
        }
    } else {
        const myBlocks = blocks.filter(b => b.player === player && !b.isFixed);
        for(let b of myBlocks) {
            let canPick = true;
            for(let c of b.cells) {
                if(c.z < 4 && board[c.z+1][c.y][c.x] !== 0) {
                    const isSelf = b.cells.some(sc=>sc.x===c.x && sc.y===c.y && sc.z===c.z+1);
                    if(!isSelf) { canPick = false; break; }
                }
            }
            if(!canPick) continue;
            const tempBoard = JSON.parse(JSON.stringify(board));
            b.cells.forEach(c => tempBoard[c.z][c.y][c.x] = 0);
            for(let i=0; i<200; i++) {
                const res = getLandingCells(tempBoard, player, i, turnCount, b.cells);
                if(res) {
                    const cSet = new Set(res.cells.map(c=>`${c.x},${c.y},${c.z}`));
                    const oSet = new Set(b.cells.map(c=>`${c.x},${c.y},${c.z}`));
                    if(cSet.size !== oSet.size || [...cSet].some(x => !oSet.has(x))) {
                        candidates.push({ type: 'move', fromId: b.id, cells: res.cells, shapeIdx: res.shapeIdx, tempBoard: tempBoard });
                    }
                }
            }
        }
    }

    // 🕵️‍♂️ 전략 1: 킬각 (내가 두면 이김?)
    for (let move of candidates) {
        let simBoard;
        if(move.type === 'place') {
            simBoard = JSON.parse(JSON.stringify(board));
            move.cells.forEach(c => simBoard[c.z][c.y][c.x] = player);
        } else {
            simBoard = JSON.parse(JSON.stringify(move.tempBoard));
            move.cells.forEach(c => simBoard[c.z][c.y][c.x] = player);
        }
        if (checkWin(simBoard) === player) return { move: move, strategy: "winning_move" };
    }

    // 🛡️ 전략 2: 방어 (상대가 두면 이김? -> 막아!)
    if (phase === 'PLACEMENT' && blocksLeft[opponent] > 0) {
        for(let i=0; i<200; i++) {
            const res = getLandingCells(board, opponent, i, turnCount);
            if(res) {
                const simBoard = JSON.parse(JSON.stringify(board));
                res.cells.forEach(c => simBoard[c.z][c.y][c.x] = opponent);
                if (checkWin(simBoard) === opponent) {
                    // 상대 킬각 발견! 내가 뺏을 수 있나?
                    const myBlock = getLandingCells(board, player, i, turnCount);
                    if (myBlock) return { move: { type: 'place', cells: myBlock.cells, shapeIdx: myBlock.shapeIdx }, strategy: "blocking_move" };
                }
            }
        }
    }

    // 🎲 전략 3: 랜덤 (임시)
    if (candidates.length > 0) return { move: candidates[Math.floor(Math.random() * candidates.length)], strategy: "random" };
    return null;
}

self.onmessage = async function(e) {
    const msg = e.data;
    if (msg.type === 'INIT') {
        try {
            neuralSession = await ort.InferenceSession.create(msg.url || './omok_model.onnx');
            useNeural = true;
            self.postMessage({ type: 'INIT_OK' });
        } catch (err) {
            self.postMessage({ type: 'INIT_FAIL', error: err.toString() });
        }
    } 
    else if (msg.type === 'THINK') {
        const { board, blocks, blocksLeft, phase, player, turnCount } = msg.gameState;
        
        let result = null;

        // 1. 뇌(ONNX) 사용 (모델이 로드되었고 Placement 단계일 때만)
        if (useNeural && phase === 'PLACEMENT') {
            try {
                const inputData = new Float32Array(1 * 3 * 5 * 5 * 5);
                const opp = player===1?2:1;
                let idx = 0;
                for(let z=0; z<5; z++) for(let y=0; y<5; y++) for(let x=0; x<5; x++) inputData[idx++] = (board[z][y][x]===player?1.0:0.0);
                for(let z=0; z<5; z++) for(let y=0; y<5; y++) for(let x=0; x<5; x++) inputData[idx++] = (board[z][y][x]===opp?1.0:0.0);
                const phaseVal = 1.0; // Placement
                for(let z=0; z<5; z++) for(let y=0; y<5; y++) for(let x=0; x<5; x++) inputData[idx++] = phaseVal;

                const tensor = new ort.Tensor('float32', inputData, [1, 3, 5, 5, 5]);
                const results = await neuralSession.run({ input: tensor });
                const logits = results.output.data;

                // ONNX가 추천한 가장 높은 점수의 '유효한' 수 찾기
                let maxScore = -Infinity;
                let bestMove = null;
                for(let i=0; i<200; i++) {
                    const res = getLandingCells(board, player, i, turnCount);
                    if(res) {
                        if(logits[i] > maxScore) {
                            maxScore = logits[i];
                            bestMove = { type: 'place', cells: res.cells, shapeIdx: res.shapeIdx };
                        }
                    }
                }
                if(bestMove) result = { move: bestMove, strategy: "neural_network" };
            } catch(e) { console.error(e); }
        }

        // 2. 뇌가 없거나, 실패했거나, Movement 단계라면 -> 스마트 계산기 가동
        if (!result) {
            result = findSmartMove(board, blocks, player, phase, blocksLeft, turnCount);
        }

        const finalMove = result ? result.move : null;
        const strategy = result ? result.strategy : "none";
        self.postMessage({ type: 'MOVE', move: finalMove, strategy: strategy });
    }
};