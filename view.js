import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class BlockOmokView {
    constructor(containerId, onHover, onClick) {
        this.container = document.getElementById(containerId);
        this.onHover = onHover;
        this.onClick = onClick;

        // 1. 씬 설정
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf5f6fa);

        // 2. 카메라
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 10, 12);
        this.camera.lookAt(0, 0, 0);

        // 3. 렌더러
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // 4. 조명
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 5);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        // 5. 컨트롤 (드래그 회전용)
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true; 
        this.controls.maxPolarAngle = Math.PI / 2; // 바닥 아래 안 보이게

        // 6. 그룹
        this.boardGroup = new THREE.Group();
        this.scene.add(this.boardGroup);
        this.ghostGroup = new THREE.Group();
        this.scene.add(this.ghostGroup);

        // 7. Raycaster
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        this.createBoard();

        // 이벤트 리스너 (드래그 vs 클릭 구분용 변수)
        let isDragging = false;
        let mouseDownTime = 0;

        this.renderer.domElement.addEventListener('mousemove', (e) => {
            isDragging = true; 
            this.onMouseMove(e);
        });

        this.renderer.domElement.addEventListener('mousedown', () => {
            isDragging = false;
            mouseDownTime = Date.now();
        });

        this.renderer.domElement.addEventListener('mouseup', (e) => {
            // 200ms 미만이고 움직임이 적었으면 '클릭'으로 간주
            const timeDiff = Date.now() - mouseDownTime;
            if (timeDiff < 200 && !isDragging) {
                this.onMouseClick(e);
            }
        });

        window.addEventListener('resize', () => this.onWindowResize());
        
        this.animate();
    }

    createBoard() {
        // 감지용 투명판 (5x5 크기 딱 맞춤)
        const planeGeo = new THREE.PlaneGeometry(5, 5);
        const planeMat = new THREE.MeshBasicMaterial({ 
            color: 0x000000, 
            transparent: true, 
            opacity: 0.1, // 흐릿하게 보여서 위치 확인 도움
            side: THREE.DoubleSide
        });
        this.hitPlane = new THREE.Mesh(planeGeo, planeMat);
        this.hitPlane.rotation.x = -Math.PI / 2;
        this.hitPlane.position.y = 0.5; // 1층 높이
        this.scene.add(this.hitPlane);

        this.scene.add(new THREE.GridHelper(5, 5, 0x888888, 0xdddddd));
    }

    onMouseMove(event) {
        // 좌표 계산
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.hitPlane);

        if (intersects.length > 0) {
            const point = intersects[0].point;
            // 좌표 변환 (엄격한 Clamp 적용)
            const gx = Math.floor(point.x + 2.5);
            const gy = Math.floor(point.z + 2.5);

            // 판 밖으로 나가면 무시
            if (gx >= 0 && gx < 5 && gy >= 0 && gy < 5) {
                this.onHover(gx, gy);
            } else {
                this.onHover(-1, -1); // 판 밖 신호
            }
        } else {
            this.onHover(-1, -1);
        }
    }

    updateGhost(cells, anchorX, anchorY, isValid) {
        while(this.ghostGroup.children.length > 0){ 
            this.ghostGroup.remove(this.ghostGroup.children[0]); 
        }

        if (anchorX === -1) return; // 커서가 밖이면 표시 안 함

        const color = isValid ? 0x2ecc71 : 0xe74c3c;
        const material = new THREE.MeshToonMaterial({ color: color, transparent: true, opacity: 0.7 });
        const geometry = new THREE.BoxGeometry(0.95, 0.95, 0.95);

        cells.forEach(cell => {
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(
                (anchorX + cell.x) - 2, 
                cell.z + 0.5, 
                (anchorY + cell.y) - 2
            );
            this.ghostGroup.add(mesh);
        });
    }

    updateBoard(boardData) {
        // 기존 블록 제거
        for (let i = this.scene.children.length - 1; i >= 0; i--) {
            if (this.scene.children[i].userData.isBlock) {
                this.scene.remove(this.scene.children[i]);
            }
        }

        const geometry = new THREE.BoxGeometry(0.98, 0.98, 0.98);
        const p1Mat = new THREE.MeshToonMaterial({ color: 0x333333 });
        const p2Mat = new THREE.MeshToonMaterial({ color: 0xffffff });

        for (let z = 0; z < 4; z++) {
            for (let y = 0; y < 5; y++) {
                for (let x = 0; x < 5; x++) {
                    const val = boardData[z][y][x];
                    if (val !== 0) {
                        const mesh = new THREE.Mesh(geometry, val === 1 ? p1Mat : p2Mat);
                        mesh.position.set(x - 2, z + 0.5, y - 2);
                        mesh.userData.isBlock = true;
                        mesh.castShadow = true;
                        mesh.receiveShadow = true;
                        this.scene.add(mesh);
                    }
                }
            }
        }
    }

    onMouseClick(event) { this.onClick(); }
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}