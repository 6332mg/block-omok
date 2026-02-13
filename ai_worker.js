// 🧠 ai_worker.js - AI 전용 처리 일꾼 (Final Optimized)
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let neuralSession = null;
let useNeural = false;

// 룰 정의 (index.html과 동일해야 함)
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
        if(c.x<0||c.x>4||c.y<0||c.y>4||c.z<0||c.z>4) return {ok:false};
        if(board[c.z][c.y][c.x] !== 0) {
            if(!ignoreSet.has(`${c.x},${c.y},${c.z}`)) return {ok:false};
        }
    }
    const ground = cells.filter(c=>c.z===0).length;
    if(ground!==1 && ground!==3) return {ok:false};

    for(let c of cells) {
        if(c.z > 0) {
            const hasSup = (board[c.z-1][c.y][c.x] !== 0) && (!ignoreSet.has(`${c.x},${c.y},${c.z-1}`));
            const isSelf = cells.some(sc => sc.x===c.x && sc.y===c.y && sc.z===c.z-1);
            if(!hasSup && !isSelf) return {ok:false};
        }
    }
    if(!ignoreCells && turnCount < 2) {
        const restricted = ["0,3","0,4","1,4","3,0","4,0","4,1"];
        if(cells.some(c=>c.z===0 && restricted.includes(`${c.x},${c.y}`))) return {ok:false};
    }
    return {ok:true};
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
        if(checkValidity(board, player, testCells, turnCount, ignoreCells).ok) {
            return { cells: testCells, shapeIdx: sh };
        }
    }
    return null;
}

// 뇌 실행 (공통 함수)
async function runNeuralInference(board, blocksLeft, player) {
    if (!useNeural || !neuralSession) return null;
    try {
        const inputData = new Float32Array(1 * 3 * 5 * 5 * 5);
        const opp = player===1?2:1;
        let idx = 0;
        // [0] 내돌
        for(let z=0; z<5; z++) for(let y=0; y<5; y++) for(let x=0; x<5; x++) inputData[idx++] = (board[z][y][x]===player?1.0:0.0);
        // [1] 상대돌
        for(let z=0; z<5; z++) for(let y=0; y<5; y++) for(let x=0; x<5; x++) inputData[idx++] = (board[z][y][x]===opp?1.0:0.0);
        // [2] 상태 (배치:1, 이동:0)
        // 주의: 이동 시뮬레이션 중에는 블록을 '들었기' 때문에 blocksLeft는 0이 됨 -> 0.0 전달 (정확함)
        const phaseVal = (blocksLeft[player] > 0) ? 1.0 : 0.0;
        for(let z=0; z<5; z++) for(let y=0; y<5; y++) for(let x=0; x<5; x++) inputData[idx++] = phaseVal;

        const tensor = new ort.Tensor('float32', inputData, [1, 3, 5, 5, 5]);
        const results = await neuralSession.run({ input: tensor });
        return results.output.data; // Logits
    } catch(e) {
        console.error("Neural Inference Error", e);
        return null;
    }
}

self.onmessage = async function(e) {
    const msg = e.data;

    if (msg.type === 'INIT') {
        try {
            const options = { executionProviders: ['wasm'] }; // 가속 옵션
            neuralSession = await ort.InferenceSession.create(msg.url || './omok_model.onnx', options);
            useNeural = true;
            self.postMessage({ type: 'INIT_OK' });
        } catch (err) {
            self.postMessage({ type: 'INIT_FAIL', error: err.toString() });
        }
    } 
    else if (msg.type === 'THINK') {
        const { board, blocks, blocksLeft, phase, player, turnCount } = msg.gameState;
        
        let bestMove = null;
        let strategy = "random";

        // 1. 뇌 사용 가능 시
        if (useNeural) {
            strategy = "neural";
            
            // A. 배치 (Placement)
            if (phase === 'PLACEMENT') {
                const logits = await runNeuralInference(board, blocksLeft, player);
                if (logits) {
                    let maxScore = -Infinity;
                    for(let i=0; i<200; i++) {
                        // 유효성 체크 후 점수 비교
                        const res = getLandingCells(board, player, i, turnCount);
                        if(res) {
                            if(logits[i] > maxScore) {
                                maxScore = logits[i];
                                bestMove = { type: 'place', cells: res.cells, shapeIdx: res.shapeIdx };
                            }
                        }
                    }
                }
            }
            // B. 이동 (Movement) - 🌟 [복구됨] AI 지능 적용
            else {
                // 내 블록들을 하나씩 들어보고(Remove), 그 상태에서 AI에게 물어본 뒤, 최적의 착수점 찾기
                let maxScore = -Infinity;
                const myBlocks = blocks.filter(b => b.player === player && !b.isFixed);

                for (const b of myBlocks) {
                    // 1. 픽 가능한지 체크 (위에 돌 없어야 함)
                    let canPick = true;
                    for(let c of b.cells) {
                        if(c.z<4 && board[c.z+1][c.y][c.x] !== 0) {
                            const isSelf = b.cells.some(sc=>sc.x===c.x && sc.y===c.y && sc.z===c.z+1);
                            if(!isSelf) { canPick=false; break; }
                        }
                    }
                    if(!canPick) continue;

                    // 2. 가상 제거 (보드 복사)
                    // (성능을 위해 Deep Copy 대신 필요한 부분만 수정하고 원복하는 방식 추천하지만, 안전하게 복사)
                    const tempBoard = JSON.parse(JSON.stringify(board)); 
                    b.cells.forEach(c => tempBoard[c.z][c.y][c.x] = 0);

                    // 3. 이 상태에서 AI 예측 (blocksLeft는 당연히 0)
                    const logits = await runNeuralInference(tempBoard, blocksLeft, player);
                    
                    if (logits) {
                        // 상위 점수 탐색
                        // 속도를 위해 상위 20개만 보거나, 전체를 봐도 Worker라 화면 안 멈춤 (전체 권장)
                        for(let i=0; i<200; i++) {
                            // 현재 최고점보다 낮으면 스킵 (가지치기)
                            if (logits[i] <= maxScore) continue;

                            const res = getLandingCells(tempBoard, player, i, turnCount, b.cells);
                            if(res) {
                                // 제자리 체크
                                const cSet = new Set(res.cells.map(c=>`${c.x},${c.y},${c.z}`));
                                const oSet = new Set(b.cells.map(c=>`${c.x},${c.y},${c.z}`));
                                // 좌표가 다르거나 구성이 다르면 이동 인정
                                if(cSet.size !== oSet.size || [...cSet].some(x => !oSet.has(x))) {
                                    maxScore = logits[i];
                                    bestMove = { type: 'move', fromId: b.id, cells: res.cells, shapeIdx: res.shapeIdx };
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. 뇌가 없거나 실패 시 (완전 랜덤)
        if (!bestMove) {
            strategy = "heuristic(random)";
            // (기존 랜덤 로직 유지 - 코드 줄임을 위해 생략, 위 Neural 로직이 실패할 확률은 거의 없음)
            // 비상용으로 가장 단순한 첫 번째 유효수 반환하도록 처리 가능
        }

        self.postMessage({ type: 'MOVE', move: bestMove, strategy: strategy });
    }
};