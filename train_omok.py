import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ============================================================================
# 🧠 [Core] 3D CNN 신경망 (공간을 입체적으로 보는 눈)
# ============================================================================
class Omok3D_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(Omok3D_CNN, self).__init__(observation_space, features_dim)
        
        # 입력 채널: 2 (내 돌, 상대 돌)
        # MX450 성능 고려: 채널 수를 32 -> 64로 적당히 조절 (너무 크면 VRAM 터짐)
        self.cnn = nn.Sequential(
            # Layer 1: 입체적 특징 추출
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Layer 2: 좀 더 복잡한 패턴 인식
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Flatten: 결정(Action)을 내리기 위해 1줄로 폄
            nn.Flatten(),
        )

        # CNN 출력 크기 계산: 64채널 * 5 * 5 * 5 = 8000
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ============================================================================
# 🏟️ [Env] 스파르타 오목 환경 (3D CNN 호환 + 하이브리드 봇)
# ============================================================================
class SpartaOmokEnv(gym.Env):
    def __init__(self):
        super(SpartaOmokEnv, self).__init__()
        # 3D CNN을 위해 Observation 형태 변경: (채널2, 높이5, 세로5, 가로5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 5, 5, 5), dtype=np.float32)
        self.action_space = spaces.Discrete(200)
        self.board_shape = (5, 5, 5)

        self.SHAPES = [
            [(0,0,0), (1,0,0), (0,1,0)], [(0,0,0), (1,0,0), (0,-1,0)],
            [(0,0,0), (-1,0,0), (0,-1,0)], [(0,0,0), (-1,0,0), (0,1,0)],
            [(0,0,0), (0,0,1), (1,0,1)], [(0,0,0), (0,0,1), (-1,0,1)],
            [(0,0,0), (0,0,1), (0,1,1)], [(0,0,0), (0,0,1), (0,-1,1)]
        ]
        self.reset()

    # 🤖 [엄격한 선생님] 그리디 70% + MCTS 30%
    def _smart_bot_turn(self):
        legal_moves = self._get_legal_moves_indices(self.opponent)
        if not legal_moves: return

        # 1. 킬각 (무조건 둠)
        for action in legal_moves:
            if self._simulate_move_fast(self.opponent, action):
                self._execute_move(self.opponent, action)
                return

        # 2. 방어 (무조건 막음)
        my_moves = self._get_legal_moves_indices(self.learner)
        threats = []
        for action in my_moves:
             if self._simulate_move_fast(self.learner, action):
                threats.append(action)
        
        for threat in threats:
            if threat in legal_moves:
                self._execute_move(self.opponent, threat)
                return

        # 3. 공격 (하이브리드 전략)
        # 30% 확률로 깊은 수읽기(MCTS), 70% 확률로 빠르고 공격적인 수(Greedy)
        if random.random() < 0.3:
            best_action = self._run_mcts_simulation_corrected(legal_moves)
        else:
            best_action = self._get_greedy_action(legal_moves)

        self._execute_move(self.opponent, best_action)

    # 🧠 [MCTS 수정판] 이제 1x1 돌이 아니라 '진짜 블록'을 랜덤으로 둬보며 시뮬레이션
    def _run_mcts_simulation_corrected(self, candidates, simulations_per_move=5, max_depth=5):
        best_score = -9999
        best_move = random.choice(candidates)

        for move in candidates:
            wins = 0
            for _ in range(simulations_per_move):
                temp_board = self.board.copy()
                
                # 가상 첫 수
                sh, px, py = move%8, (move//8)%5, (move//8)//5
                cells = self._get_cells(px, py, sh)
                for c in cells: temp_board[c['z']][c['y']][c['x']] = self.opponent
                
                sim_turn = 0
                current_sim_player = self.learner 
                my_sim_id = self.opponent
                
                while sim_turn < max_depth:
                    if self._check_win_simulation(temp_board) == my_sim_id:
                        wins += 1; break
                    
                    # 랜덤 착수 (유효한 것 찾을 때까지 최대 10번 시도)
                    placed = False
                    for _ in range(10):
                        r_idx = random.randint(0, 199)
                        r_sh, r_px, r_py = r_idx%8, (r_idx//8)%5, (r_idx//8)//5
                        r_cells = self._get_cells(r_px, r_py, r_sh)
                        if self._check_validity_simple_for_sim(temp_board, r_cells):
                            for c in r_cells: temp_board[c['z']][c['y']][c['x']] = current_sim_player
                            placed = True
                            break
                    if not placed: break 

                    current_sim_player = my_sim_id if current_sim_player != my_sim_id else (3 - my_sim_id)
                    sim_turn += 1
            
            if wins > best_score:
                best_score = wins
                best_move = move
        return best_move

    # 🔥 [Greedy 전략] 님의 로직 (중앙, 높이, 인접 가산점)
    def _get_greedy_action(self, candidates):
        best_action = -1
        max_score = -9999
        for action in candidates:
            score = 0
            sh, px, py = action%8, (action//8)%5, (action//8)//5
            cells = self._get_cells(px, py, sh)
            
            for c in cells:
                score += (2 - abs(c['x'] - 2)) + (2 - abs(c['y'] - 2)) # 중앙
                score += (4 - c['z']) * 0.5 # 낮은 높이 선호
                # 인접 체크
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    nx, ny, nz = c['x']+dx, c['y']+dy, c['z']+dz
                    if 0<=nx<5 and 0<=ny<5 and 0<=nz<5:
                        if self.board[nz][ny][nx] == self.opponent: score += 1.5 
            
            score += random.uniform(0, 1.0) # 약간의 랜덤성
            if score > max_score:
                max_score = score
                best_action = action
        return best_action

    # 시뮬레이션용 초간단 유효성 체크 (속도 최우선)
    def _check_validity_simple_for_sim(self, board, cells):
         for c in cells:
             if not (0<=c['x']<5 and 0<=c['y']<5 and 0<=c['z']<5): return False
             if board[c['z']][c['y']][c['x']] != 0: return False
         return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(self.board_shape, dtype=np.int8)
        self.blocks = []
        self.turn_count = 0
        self.blocks_left = {1: 4, 2: 4}
        self.phase = 'PLACEMENT'

        # 초기 배치 (고정)
        self._add_block(1, [{'x':1,'y':3,'z':0}, {'x':2,'y':3,'z':0}, {'x':1,'y':2,'z':0}], 0, True)
        self._add_block(2, [{'x':2,'y':1,'z':0}, {'x':3,'y':1,'z':0}, {'x':3,'y':2,'z':0}], 0, True)

        self.learner = random.choice([1, 2])
        self.opponent = 2 if self.learner == 1 else 1
        self.current_player = 1

        if self.opponent == 1:
            self._smart_bot_turn()

        return self._get_obs(), {}

    def _get_obs(self):
        # 3D CNN용 Observation: (Channel, Depth, Height, Width)
        # Channel 0: 내 돌, Channel 1: 상대 돌
        my_stones = (self.board == self.learner).astype(np.float32)
        opp_stones = (self.board == self.opponent).astype(np.float32)
        return np.stack([my_stones, opp_stones], axis=0) # shape: (2, 5, 5, 5)

    def step(self, action):
        if not self._execute_move(self.learner, action):
            # 룰 위반 시 강력한 페널티
            return self._get_obs(), -50, True, False, {}

        if self._check_win() == self.learner:
            return self._get_obs(), 100, True, False, {}

        self._next_turn()

        # 봇 착수
        self._smart_bot_turn()

        if self._check_win() == self.opponent:
            # 지면 매우 큰 페널티 (4목 허용 방지)
            return self._get_obs(), -500, True, False, {}

        self._next_turn()

        terminated = False
        if self.turn_count > 100: terminated = True
        return self._get_obs(), -0.1, terminated, False, {}

    # --- (기존 로직 유지) ---
    def _simulate_move_fast(self, player, action):
        sh, px, py = action%8, (action//8)%5, (action//8)//5
        cells = self._get_cells(px, py, sh)
        if not self._check_validity_simple(player, cells): return False
        for c in cells: self.board[c['z']][c['y']][c['x']] = player
        win = (self._check_win() == player)
        for c in cells: self.board[c['z']][c['y']][c['x']] = 0
        return win

    def _get_legal_moves_indices(self, player):
        mask = self._get_action_masks_for_player(player)
        return np.where(mask)[0].tolist()

    def _next_turn(self):
        self.turn_count += 1
        self.current_player = 2 if self.current_player == 1 else 1
        if self.blocks_left[1] == 0 and self.blocks_left[2] == 0: self.phase = 'MOVEMENT'
        elif self.blocks_left[self.current_player] == 0 and self.phase == 'PLACEMENT': self.phase = 'MOVEMENT'

    def _execute_move(self, player, action):
        sh, px, py = action%8, (action//8)%5, (action//8)//5
        cells = self._get_cells(px, py, sh)
        if self.phase == 'PLACEMENT':
            if self._check_validity_simple(player, cells):
                self._add_block(player, cells, sh)
                self.blocks_left[player] -= 1
                return True
        else:
            my_blocks = [b for b in self.blocks if b['player'] == player and not b.get('fixed')]
            for b in my_blocks:
                if not self._can_pick(b): continue
                orig = b['cells']
                self._remove_block(b['id'])
                if self._check_validity_simple(player, cells, True, orig):
                    self._add_block(player, cells, sh)
                    return True
                self._add_block(player, orig, b['shapeIdx'], False, b['id'])
        return False

    def action_masks(self): return self._get_action_masks_for_player(self.learner)

    def _get_action_masks_for_player(self, player):
        mask = np.zeros(200, dtype=bool)
        target_phase = self.phase
        if target_phase == 'PLACEMENT':
            if self.blocks_left[player] > 0:
                for i in range(200):
                    sh, px, py = i%8, (i//8)%5, (i//8)//5
                    cells = self._get_cells(px, py, sh)
                    if self._check_validity_simple(player, cells): mask[i] = True
        else:
            my_blocks = [b for b in self.blocks if b['player'] == player and not b.get('fixed') and self._can_pick(b)]
            for b in my_blocks:
                orig = b['cells']
                self._remove_block(b['id'])
                for i in range(200):
                    if mask[i]: continue
                    sh, px, py = i%8, (i//8)%5, (i//8)//5
                    cells = self._get_cells(px, py, sh)
                    if self._check_validity_simple(player, cells, True, orig): mask[i] = True
                self._add_block(player, orig, b['shapeIdx'], False, b['id'])
        return mask

    def _get_cells(self, bx, by, shape_idx):
        shape = self.SHAPES[shape_idx]
        return [{'x': bx + dx, 'y': by + dy, 'z': dz} for dx, dy, dz in shape]
    def _add_block(self, player, cells, shape_idx, is_fixed=False, block_id=None):
        if block_id is None: block_id = self.turn_count * 10000 + len(self.blocks)
        self.blocks.append({'id': block_id, 'player': player, 'cells': cells, 'shapeIdx': shape_idx, 'fixed': is_fixed})
        for c in cells: self.board[c['z']][c['y']][c['x']] = player
    def _remove_block(self, block_id):
        idx = next((i for i, b in enumerate(self.blocks) if b['id'] == block_id), -1)
        if idx != -1:
            block = self.blocks.pop(idx)
            for c in block['cells']: self.board[c['z']][c['y']][c['x']] = 0
    def _can_pick(self, block):
        if block.get('fixed'): return False
        for c in block['cells']:
            if c['z'] >= 4: continue
            if self.board[c['z']+1][c['y']][c['x']] != 0:
                is_self = any(sc['x']==c['x'] and sc['y']==c['y'] and sc['z']==c['z']+1 for sc in block['cells'])
                if not is_self: return False
        return True
    def _check_validity_simple(self, player, cells, is_movement=False, original_cells=None):
        for c in cells:
            if not (0<=c['x']<5 and 0<=c['y']<5 and 0<=c['z']<5): return False
            if self.board[c['z']][c['y']][c['x']] != 0: return False
        if is_movement and original_cells:
            c_set = set((c['x'],c['y'],c['z']) for c in cells)
            o_set = set((c['x'],c['y'],c['z']) for c in original_cells)
            if c_set == o_set: return False
        ground = sum(1 for c in cells if c['z']==0)
        if ground != 3 and ground != 1: return False
        for c in cells:
            if c['z'] > 0:
                has_sup = self.board[c['z']-1][c['y']][c['x']] != 0
                is_self = any(sc['x']==c['x'] and sc['y']==c['y'] and sc['z']==c['z']-1 for sc in cells)
                if not has_sup and not is_self: return False
        if not is_movement and self.turn_count < 2:
            restricted = ["0,3", "0,4", "1,4", "3,0", "4,0", "4,1"]
            for c in cells:
                if c['z']==0 and f"{c['x']},{c['y']}" in restricted: return False
        return True
    def _check_win(self):
        top_map = np.zeros((5,5), dtype=int)
        for y in range(5):
            for x in range(5):
                for z in range(4, -1, -1):
                    if self.board[z][y][x] != 0: top_map[y][x] = self.board[z][y][x]; break
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for y in range(5):
            for x in range(5):
                c = top_map[y][x]
                if c == 0: continue
                for dx, dy in dirs:
                    cnt = 1
                    for k in range(1, 5):
                        nx, ny = x+dx*k, y+dy*k
                        if 0<=nx<5 and 0<=ny<5 and top_map[ny][nx]==c: cnt+=1
                        else: break
                    if cnt == 5: return c
        return 0
    # 시뮬레이션용 (Board를 인자로 받음)
    def _check_win_simulation(self, board_arr):
        top_map = np.zeros((5,5), dtype=int)
        for y in range(5):
            for x in range(5):
                for z in range(4, -1, -1):
                    if board_arr[z][y][x] != 0: top_map[y][x] = board_arr[z][y][x]; break
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for y in range(5):
            for x in range(5):
                c = top_map[y][x]
                if c == 0: continue
                for dx, dy in dirs:
                    cnt = 1
                    for k in range(1, 5):
                        nx, ny = x+dx*k, y+dy*k
                        if 0<=nx<5 and 0<=ny<5 and top_map[ny][nx]==c: cnt+=1
                        else: break
                    if cnt == 5: return c
        return 0

# ============================================================================
# 💾 [Local Save] 저장 콜백
# ============================================================================
class LocalSaveCallback(BaseCallback):
    def __init__(self, save_freq=100000, save_path="./models", verbose=0):
        super(LocalSaveCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.gen_count = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.gen_count += 1
            path = os.path.join(self.save_path, f"sparta_cnn_gen_{self.gen_count}")
            self.model.save(path)
            if self.verbose > 0:
                print(f"💾 [Local] CNN Model Generation {self.gen_count} Saved! (Step: {self.num_timesteps})")
        return True

def mask_fn(env): return env.get_wrapper_attr('action_masks')()

# ============================================================================
# 🏃‍♂️ [Main] 실행부 (GPU 가속 + 3D CNN)
# ============================================================================
if __name__ == '__main__':
    # i5-1135G7 (4코어) 고려하여 환경 4개 병렬 처리
    n_envs = 4 
    
    # GPU 강제 할당 확인
    if torch.cuda.is_available():
        device = "cuda"
        # MX450은 메모리가 작으므로 캐시 정리 한 번 해줌
        torch.cuda.empty_cache()
        print(f"🖥️ NVIDIA GeForce MX450 가동! (CUDA Available)")
    else:
        device = "cpu"
        print("⚠️ GPU를 찾을 수 없습니다. CPU로 실행합니다.")

    # 환경 생성
    vec_env = SubprocVecEnv([lambda: ActionMasker(SpartaOmokEnv(), mask_fn) for _ in range(n_envs)])

    # 이어하기 체크
    load_filename = "sparta_cnn_final.zip"
    
    # 🔥 [3D CNN 정책 설정]
    # CnnPolicy를 쓰되, features_extractor_class를 우리가 만든 3D CNN으로 교체
    policy_kwargs = dict(
        features_extractor_class=Omok3D_CNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[] # CNN에서 나온 256개 특징을 바로 Action Net으로 (MLP 추가 안 함)
    )

    if os.path.exists(load_filename):
        print(f"♻️ '{load_filename}' 발견! 훈련을 이어갑니다...")
        model = MaskablePPO.load(load_filename, env=vec_env, device=device)
    else:
        print("✨ 3D CNN을 탑재한 새로운 AI가 태어납니다!")
        model = MaskablePPO(
            "CnnPolicy", # 3D CNN을 쓰더라도 베이스는 CnnPolicy
            vec_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024, # VRAM 절약을 위해 2048 -> 1024로 약간 줄임
            batch_size=128, # VRAM 절약을 위해 256 -> 128로 줄임 (MX450 최적화)
            gamma=0.99,
            device=device,
            policy_kwargs=policy_kwargs # 커스텀 3D CNN 주입
        )

    print("🔥 [Sparta 3D] 지옥 훈련 시작! (Hybrid Opponent: Greedy + MCTS) 🔥")
    
    total_steps = 3000000 # 300만번
    callback = LocalSaveCallback(save_freq=50000, save_path="./models", verbose=1)

    try:
        model.learn(total_timesteps=total_steps, callback=callback)
        model.save("sparta_cnn_final")
        print("✅ 훈련 완료. sparta_cnn_final.zip 저장됨.")
    except KeyboardInterrupt:
        print("\n🛑 중단됨. 현재 상태 저장 중...")
        model.save("sparta_cnn_interrupted")
        print("✅ 저장 완료.")