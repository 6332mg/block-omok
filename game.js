import { BlockOmokLogic } from './logic.js';
import { BlockOmokView } from './view.js';

const logic = new BlockOmokLogic();
const logEl = document.getElementById('console-log');

let currentShapeIndex = 0; 
let cursorX = -1;
let cursorY = -1;

// [중요] ㄱ자 블록 8가지 모양 정의
// 기준점(0,0,0)을 중심으로 상대 좌표
const SHAPES = [
    // --- 1. 바닥에 누운 모양 (4개) ---
    [{x:0, y:0, z:0}, {x:1, y:0, z:0}, {x:0, y:1, z:0}],  // 기본 ㄴ
    [{x:0, y:0, z:0}, {x:1, y:0, z:0}, {x:0, y:-1, z:0}], // 90도
    [{x:0, y:0, z:0}, {x:-1, y:0, z:0}, {x:0, y:-1, z:0}],// 180도
    [{x:0, y:0, z:0}, {x:-1, y:0, z:0}, {x:0, y:1, z:0}], // 270도
    
    // --- 2. 세워진 모양 (4개) - z=1 포함 ---
    // 기준점 위로 하나가 솟아오른 형태
    [{x:0, y:0, z:0}, {x:1, y:0, z:0}, {x:0, y:0, z:1}], // 누운거 하나가 위로 감
    [{x:0, y:0, z:0}, {x:-1, y:0, z:0}, {x:0, y:0, z:1}],
    [{x:0, y:0, z:0}, {x:0, y:1, z:0}, {x:0, y:0, z:1}],
    [{x:0, y:0, z:0}, {x:0, y:-1, z:0}, {x:0, y:0, z:1}]
];

function log(msg) {
    if(logEl) logEl.innerText = msg + "\n" + logEl.innerText;
}

const view = new BlockOmokView('game-canvas', 
    (x, y) => { // onHover
        cursorX = x;
        cursorY = y;
        updateGhost();
    },
    () => { // onClick
        tryPlaceBlock();
    }
);

function updateGhost() {
    if (cursorX === -1) {
        view.updateGhost([], -1, -1, false); // 숨김
        return;
    }

    const shapeTemplate = SHAPES[currentShapeIndex % 8];
    // 현재 커서 위치 기준 절대 좌표 계산
    const cells = shapeTemplate.map(offset => ({
        x: cursorX + offset.x,
        y: cursorY + offset.y,
        z: offset.z // 기본 z (세워진 블록은 여기서 z=1이 됨)
        // 참고: 'Space'로 2층 착수를 구현하려면 여기에 + targetZ 추가
    }));

    const check = logic.checkPlacementValidity(logic.currentPlayer, cells);
    view.updateGhost(cells, cursorX, cursorY, check.valid);
}

function tryPlaceBlock() {
    if (cursorX === -1) return;
    
    const shapeTemplate = SHAPES[currentShapeIndex % 8];
    const cells = shapeTemplate.map(offset => ({
        x: cursorX + offset.x,
        y: cursorY + offset.y,
        z: offset.z
    }));

    const check = logic.checkPlacementValidity(logic.currentPlayer, cells);

    if (check.valid) {
        logic.placeBlockInternal(logic.currentPlayer, cells, `block_${Date.now()}`);
        logic.turnCount++;
        view.updateBoard(logic.board);
        
        const winner = logic.checkWin();
        if (winner !== 0) {
            alert(`Player ${winner} 승리!`);
        } else {
            logic.currentPlayer = logic.currentPlayer === 1 ? 2 : 1;
            document.getElementById('current-player').innerText = 
                logic.currentPlayer === 1 ? "흑(Player 1)" : "백(Player 2)";
        }
    } else {
        log(`[불가] ${check.message}`);
    }
}

// 키보드 조작
window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyR') {
        currentShapeIndex++;
        const type = (currentShapeIndex % 8) < 4 ? "눕힘" : "세움";
        log(`회전: 모양 ${currentShapeIndex % 8 + 1} (${type})`);
        updateGhost();
    }
});

// 초기화
view.updateBoard(logic.board);
log("게임 시작. 드래그=화면회전, R=블록회전, 클릭=착수");