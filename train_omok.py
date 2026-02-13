import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ============================================================================
# 🧬 [Core] 3D CNN (피드백 반영: 3채널 유지)
# ============================================================================
class Omok3D_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(Omok3D_CNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ============================================================================
# 🏟️ [Env] 스파르타 오목 환경 (Final Fix Ver)
# ============================================================================
class SpartaOmokEnv(gym.Env):
    def __init__(self):
        super(SpartaOmokEnv, self).__init__()
        # [0:내돌, 1:상대돌, 2:페이즈]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 5, 5, 5), dtype=np.float32)
        self.action_space = spaces.Discrete(200)
        self.board_shape = (5, 5, 5)

        self.SHAPES = [
            [(0,0,0), (1,0,0), (0,1,0)], [(0,0,0), (1,0,0), (0,-1,0)],
            [(0,0,0), (-1,0,0), (0,-1,0)], [(0,0,0), (-1,0,0), (0,1,0)],
            [(0,0,0), (0,0,1), (1,0,1)], [(0,0,0), (0,0,1), (-1,0,1)],
            [(0,0,0), (0,0,1), (0,1,1)], [(0,0,0), (0,0,1), (0,-1,1)]
        ]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(self.board_shape, dtype=np.int8)
        self.blocks = []
        self.turn_count = 0
        self.blocks_left = {1: 4, 2: 4}
        self.phase = 'PLACEMENT'

        # 고정 블록 배치
        self._add_block(1, [{'x':1,'y':3,'z':0}, {'x':2,'y':3,'z':0}, {'x':1,'y':2,'z':0}], 0, True)
        self._add_block(2, [{'x':2,'y':1,'z':0}, {'x':3,'y':1,'z':0}, {'x':3,'y':2,'z':0}], 0, True)

        self.learner = random.choice([1, 2])
        self.opponent = 2 if self.learner == 1 else 1
        self.current_player = 1
        
        # 🔧 [수정] 초기 상태 점수 계산 (이게 0이면 첫 수에 점수가 튀어버림)
        self.prev_score = self._evaluate_board(self.learner)

        if self.opponent == 1: self._smart_bot_turn()
        return self._get_obs(), {}

    def step(self, action):
        # 1. AI 착수
        if not self._execute_move(self.learner, action):
            # 룰 위반 시 페널티 (학습 초기에만 발생하고 사라짐)
            return self._get_obs(), -5.0, True, False, {} 

        # 승리 체크
        if self._check_win() == self.learner:
            return self._get_obs(), 100.0, True, False, {}

        # 보상 쉐이핑
        current_score = self._evaluate_board(self.learner)
        shaping_reward = (current_score - self.prev_score) * 0.1
        self.prev_score = current_score

        self._next_turn()

        # 2. 상대 봇 착수
        self._smart_bot_turn()

        # 패배 체크
        if self._check_win() == self.opponent:
            return self._get_obs(), -100.0, True, False, {}

        self._next_turn()
        
        terminated = False
        if self.turn_count > 100: terminated = True

        return self._get_obs(), shaping_reward - 0.01, terminated, False, {}

    # --- 평가 함수 ---
    def _evaluate_board(self, player):
        top_map = np.zeros((5,5), dtype=int)
        for y in range(5):
            for x in range(5):
                for z in range(4, -1, -1):
                    if self.board[z][y][x] != 0: top_map[y][x] = self.board[z][y][x]; break
        
        score = 0
        opp = 3 - player
        
        lines = []
        for y in range(5): lines.append(top_map[y, :])
        for x in range(5): lines.append(top_map[:, x])
        lines.append(np.diag(top_map)); lines.append(np.diag(np.fliplr(top_map)))

        for line in lines:
            line_list = line.tolist()
            score += self._count_pattern(line_list, player, 4) * 20
            score += self._count_pattern(line_list, player, 3) * 10
            score -= self._count_pattern(line_list, opp, 4) * 25
            score -= self._count_pattern(line_list, opp, 3) * 15
        return score

    def _count_pattern(self, line, player, count):
        cnt = 0; consecutive = 0
        for cell in line:
            if cell == player: consecutive += 1
            else:
                if consecutive >= count: cnt += 1
                consecutive = 0
        if consecutive >= count: cnt += 1
        return cnt

    # --- 🤖 스마트 봇 (버그 수정됨) ---
    def _smart_bot_turn(self):
        mask = self._get_action_masks_for_player(self.opponent)
        legal_moves = np.where(mask)[0].tolist()
        if not legal_moves: return

        # 1. 킬각
        for action in legal_moves:
            if self._simulate_win(self.opponent, action):
                self._execute_move(self.opponent, action); return
        
        # 2. 최선의 수 찾기 (Greedy Simulation)
        best_action = -1
        max_eval = -99999
        candidates = random.sample(legal_moves, min(len(legal_moves), 15))
        
        for action in candidates:
            # 🔧 [수정] 중요 상태 백업 (blocks_left, phase 추가!)
            backup_board = self.board.copy()
            backup_blocks = [b.copy() for b in self.blocks]
            backup_left = self.blocks_left.copy() # ⭐️ 필수 복구 대상
            backup_phase = self.phase             # ⭐️ 필수 복구 대상
            
            # 가상 착수
            success = self._execute_move(self.opponent, action)
            
            if success:
                eval_score = self._evaluate_board(self.opponent)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
            
            # 🔧 [수정] 완벽 복구 (Rollback)
            self.board = backup_board
            self.blocks = backup_blocks
            self.blocks_left = backup_left
            self.phase = backup_phase
            
        final_action = best_action if best_action != -1 else random.choice(legal_moves)
        self._execute_move(self.opponent, final_action)

    # --- 물리 엔진 & 실행 ---
    def _execute_move(self, player, action):
        if self.phase == 'PLACEMENT':
            cells, sh = self._get_landing_cells(action, player)
            if cells:
                self._add_block(player, cells, sh)
                self.blocks_left[player] -= 1
                return True
            return False
        else:
            my_blocks = [b for b in self.blocks if b['player'] == player and not b.get('fixed')]
            target_sh = action % 8
            
            for b in my_blocks:
                if not self._can_pick(b): continue
                orig_cells = b['cells']
                # 가상 제거
                for c in orig_cells: self.board[c['z']][c['y']][c['x']] = 0
                
                # 착지점 계산
                landing_cells, _ = self._get_landing_cells(action, player, ignore_block_cells=orig_cells)
                
                if landing_cells:
                    c_set = set((c['x'],c['y'],c['z']) for c in landing_cells)
                    o_set = set((c['x'],c['y'],c['z']) for c in orig_cells)
                    if c_set != o_set:
                        self._remove_block_internal(b['id'])
                        self._add_block(player, landing_cells, target_sh, block_id=b['id'])
                        return True
                # 실패 시 복구
                for c in orig_cells: self.board[c['z']][c['y']][c['x']] = player
            return False

    def _get_landing_cells(self, action_idx, player, ignore_block_cells=None):
        sh, px, py = action_idx % 8, (action_idx // 8) % 5, (action_idx // 8) // 5
        base_shape = self.SHAPES[sh]
        # 중력 시뮬레이션: 바닥부터 훑어서 가장 낮은 valid 위치 리턴
        for dz in range(5):
            test_cells = [{'x': px + dx, 'y': py + dy, 'z': dz + d_z_offset} for dx, dy, d_z_offset in base_shape]
            if any(c['z'] > 4 for c in test_cells): break
            if self._check_validity_strict(player, test_cells, ignore_block_cells): return test_cells, sh
        return None, None

    def _check_validity_strict(self, player, cells, ignore_block_cells=None):
        ignore_set = set()
        if ignore_block_cells:
            for c in ignore_block_cells: ignore_set.add((c['x'], c['y'], c['z']))
            
        # 1. 충돌 체크
        for c in cells:
            if not (0<=c['x']<5 and 0<=c['y']<5 and 0<=c['z']<5): return False
            if self.board[c['z']][c['y']][c['x']] != 0:
                if (c['x'], c['y'], c['z']) not in ignore_set: return False
                
        # 2. 제자리 체크
        if ignore_block_cells:
            if set((c['x'], c['y'], c['z']) for c in cells) == ignore_set: return False
            
        # 3. 지지대 체크
        ground_contact = sum(1 for c in cells if c['z'] == 0)
        for c in cells:
            if c['z'] > 0:
                below_x, below_y, below_z = c['x'], c['y'], c['z']-1
                has_support = (self.board[below_z][below_y][below_x] != 0)
                if (below_x, below_y, below_z) in ignore_set: has_support = False # 내가 있던 자리는 지지대 아님
                is_self_support = any(sc['x']==below_x and sc['y']==below_y and sc['z']==below_z for sc in cells)
                if not has_support and not is_self_support: return False
                
        # 4. 바닥 닿는 면적 규칙
        if ground_contact > 0 and ground_contact not in [1, 3]: return False
        
        # 5. 초반 금지 구역
        if not ignore_block_cells and self.turn_count < 2:
            restricted = ["0,3", "0,4", "1,4", "3,0", "4,0", "4,1"]
            for c in cells:
                if c['z'] == 0 and f"{c['x']},{c['y']}" in restricted: return False
        return True

    def _can_pick(self, block):
        if block.get('fixed'): return False
        for c in block['cells']:
            if c['z'] >= 4: continue
            if self.board[c['z']+1][c['y']][c['x']] != 0:
                is_self_part = any(sc['x']==c['x'] and sc['y']==c['y'] and sc['z']==c['z']+1 for sc in block['cells'])
                if not is_self_part: return False
        return True

    def _add_block(self, player, cells, shape_idx, is_fixed=False, block_id=None):
        if block_id is None: block_id = self.turn_count * 10000 + len(self.blocks)
        self.blocks.append({'id': block_id, 'player': player, 'cells': cells, 'shapeIdx': shape_idx, 'fixed': is_fixed})
        for c in cells: self.board[c['z']][c['y']][c['x']] = player

    def _remove_block_internal(self, block_id):
        idx = next((i for i, b in enumerate(self.blocks) if b['id'] == block_id), -1)
        if idx != -1: self.blocks.pop(idx)

    def _simulate_win(self, player, action):
        backup_board = self.board.copy()
        backup_blocks = [b.copy() for b in self.blocks]
        backup_left = self.blocks_left.copy()
        backup_phase = self.phase 
        
        win = False
        if self._execute_move(player, action):
            if self._check_win() == player: win = True
            
        self.board = backup_board
        self.blocks = backup_blocks
        self.blocks_left = backup_left
        self.phase = backup_phase
        return win

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
    
    def _get_obs(self):
        my_stones = (self.board == self.learner).astype(np.float32)
        opp_stones = (self.board == self.opponent).astype(np.float32)
        phase_val = 1.0 if self.phase == 'PLACEMENT' else 0.0
        state_channel = np.full(self.board_shape, phase_val, dtype=np.float32)
        return np.stack([my_stones, opp_stones, state_channel], axis=0)
    
    def action_masks(self): return self._get_action_masks_for_player(self.learner)
    def _get_action_masks_for_player(self, player):
        mask = np.zeros(200, dtype=bool)
        if self.phase == 'PLACEMENT':
            if self.blocks_left[player] > 0:
                for i in range(200):
                    cells, _ = self._get_landing_cells(i, player)
                    if cells: mask[i] = True
        else:
            my_blocks = [b for b in self.blocks if b['player'] == player and not b.get('fixed') and self._can_pick(b)]
            for i in range(200):
                possible = False
                for b in my_blocks:
                    # 마스킹 연산량 줄이기: _get_landing_cells 호출 전에 충돌 체크
                    for c in b['cells']: self.board[c['z']][c['y']][c['x']] = 0
                    l_cells, _ = self._get_landing_cells(i, player, ignore_block_cells=b['cells'])
                    for c in b['cells']: self.board[c['z']][c['y']][c['x']] = player
                    if l_cells:
                         c_set = set((c['x'],c['y'],c['z']) for c in l_cells)
                         o_set = set((c['x'],c['y'],c['z']) for c in b['cells'])
                         if c_set != o_set: possible = True; break
                if possible: mask[i] = True
        return mask

    def _get_legal_moves_indices(self, player): return np.where(self._get_action_masks_for_player(player))[0].tolist()

class LocalSaveCallback(BaseCallback):
    def __init__(self, save_freq=50000, save_path="./models", verbose=0):
        super(LocalSaveCallback, self).__init__(verbose)
        self.save_freq = save_freq; self.save_path = save_path; self.gen_count = 0; os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.gen_count += 1
            self.model.save(os.path.join(self.save_path, f"sparta_final_gen_{self.gen_count}"))
        return True

def mask_fn(env): return env.get_wrapper_attr('action_masks')()

if __name__ == '__main__':
    n_envs = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device.upper()}")
    vec_env = SubprocVecEnv([lambda: ActionMasker(SpartaOmokEnv(), mask_fn) for _ in range(n_envs)])
    
    # 3채널 CNN
    policy_kwargs = dict(features_extractor_class=Omok3D_CNN, features_extractor_kwargs=dict(features_dim=256), net_arch=[])
    
    model = MaskablePPO("CnnPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=1024, batch_size=128, gamma=0.99, device=device, policy_kwargs=policy_kwargs)
    print("🔥 [Final Ver] 시뮬레이션 완벽 복구 & 보상 쉐이핑. 훈련 시작! 🔥")
    
    try:
        model.learn(total_timesteps=3000000, callback=LocalSaveCallback(save_freq=50000, save_path="./models", verbose=1))
        model.save("sparta_final_complete")
        print("✅ 완료.")
    except KeyboardInterrupt:
        model.save("sparta_final_interrupted")
        print("✅ 저장 완료.")