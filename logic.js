export class BlockOmokLogic {
    constructor() {
        // 5x5x4 보드 (높이 여유 있게 4까지)
        this.board = Array.from({ length: 4 }, () => 
            Array.from({ length: 5 }, () => Array(5).fill(0))
        );
        this.turnCount = 0; 
        this.currentPlayer = 1; 
        this.initFixedBlocks();
    }

    initFixedBlocks() {
        // 흑(P1) 고정: (2,4),(3,4),(2,3) -> idx [1,3], [2,3], [1,2]
        this.placeBlockInternal(1, [{x:1, y:3, z:0}, {x:2, y:3, z:0}, {x:1, y:2, z:0}], 'p1_fix');
        // 백(P2) 고정: (3,2),(4,2),(4,3) -> idx [2,1], [3,1], [3,2]
        this.placeBlockInternal(2, [{x:2, y:1, z:0}, {x:3, y:1, z:0}, {x:3, y:2, z:0}], 'p2_fix');
    }

    placeBlockInternal(player, cells, blockId) {
        cells.forEach(cell => {
            this.board[cell.z][cell.y][cell.x] = player;
        });
    }

    checkPlacementValidity(player, cells) {
        // 1. 범위 및 중복 검사
        for (let c of cells) {
            if (c.x < 0 || c.x > 4 || c.y < 0 || c.y > 4 || c.z < 0) {
                return { valid: false, message: "판을 벗어났습니다." };
            }
            if (this.board[c.z][c.y][c.x] !== 0) {
                return { valid: false, message: "이미 블록이 있습니다." };
            }
        }

        // 2. 금지 구역 (초반 2턴: 흑0, 백0)
        if (this.turnCount < 2) {
            // (1,4)(1,5)(2,5) -> (0,3)(0,4)(1,4)
            // (4,1)(5,1)(5,2) -> (3,0)(4,0)(4,1)
            const restricted = [
                "0,3", "0,4", "1,4", 
                "3,0", "4,0", "4,1"
            ];
            
            for (let c of cells) {
                if (c.z === 0 && restricted.includes(`${c.x},${c.y}`)) {
                    return { valid: false, message: "초반 2턴 금지 구역입니다." };
                }
            }
        }

        // 3. 물리 규칙 (바닥 지지 & 공중부양 방지)
        const touchesGround = cells.some(c => c.z === 0);
        let supported = true;
        
        for (let c of cells) {
            // 2층 이상인 경우 바로 아래에 블록이 있어야 함
            if (c.z > 0) {
                // 바로 아래칸이 비어있고, 자기 자신의 다른 파트도 아니라면 지지대 없음
                // (자기 자신의 다른 파트가 아래에 있는 경우는 '수직으로 선' 형태)
                const isSelfSupport = cells.some(sc => sc.x === c.x && sc.y === c.y && sc.z === c.z - 1);
                if (this.board[c.z - 1][c.y][c.x] === 0 && !isSelfSupport) {
                    supported = false;
                }
            }
        }

        if (!touchesGround) return { valid: false, message: "적어도 한 칸은 바닥(1층)에 닿아야 합니다." };
        if (!supported) return { valid: false, message: "블록 아래에 지지대가 필요합니다." };

        return { valid: true, message: "가능" };
    }

    checkWin() {
        let topMap = Array.from({ length: 5 }, () => Array(5).fill(0));
        // Top View 생성 (가장 높은 블록 색깔)
        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 5; x++) {
                for (let z = 3; z >= 0; z--) {
                    if (this.board[z][y][x] !== 0) {
                        topMap[y][x] = this.board[z][y][x];
                        break;
                    }
                }
            }
        }
        
        // 5목 체크 (전체 스캔)
        const dirs = [{dx:1, dy:0}, {dx:0, dy:1}, {dx:1, dy:1}, {dx:1, dy:-1}];
        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 5; x++) {
                let color = topMap[y][x];
                if (color === 0) continue;
                for (let d of dirs) {
                    let k;
                    for (k = 1; k < 5; k++) {
                        let nx = x + d.dx * k, ny = y + d.dy * k;
                        if (nx < 0 || nx >= 5 || ny < 0 || ny >= 5 || topMap[ny][nx] !== color) break;
                    }
                    if (k === 5) return color;
                }
            }
        }
        return 0;
    }
}