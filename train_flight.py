import os
import sys
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ============================================================================
# ⚙️ [비행기 모드 설정]
# ============================================================================
N_ENVS = 2          # 노트북 사양에 맞게 조정 (너무 느리면 1로 변경)
DEVICE = "cpu"      # 배터리 절약을 위해 CPU 권장
LOAD_MODEL_NAME = "sparta_colab_gen_10.zip" # 같은 폴더에 있어야 함
SAVE_FREQ = 5000    # 저장 주기 (약 10~15분마다 저장되도록 단축)
SAVE_DIR = "./flight_models" # 결과물이 저장될 폴더
# ============================================================================

# 파일 경로 확인 (비행기에서 당황하지 않기 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, LOAD_MODEL_NAME)

print("="*50)
print(f"✈️ 비행기 모드 학습 준비")
print(f"📂 작업 폴더: {current_dir}")
print(f"📂 모델 찾기: {model_path}")

if not os.path.exists(model_path):
    print("\n🚨 [치명적 오류] 모델 파일이 없습니다!")
    print(f"'{LOAD_MODEL_NAME}' 파일을 이 스크립트와 같은 폴더에 넣어주세요.")
    print("비행기 타기 전에 꼭 확인하세요!")
    input("엔터를 누르면 종료합니다...")
    sys.exit()
else:
    print("✅ 모델 파일 확인 완료. 로딩을 준비합니다.")
print("="*50)

# ============================================================================
# 🧬 AI 모델 및 환경 (로직 동일)
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

class SpartaOmokEnv(gym.Env):
    def __init__(self):
        super(SpartaOmokEnv, self).__init__()
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
        self._add_block(1, [{'x':1,'y':3,'z':0}, {'x':2,'y':3,'z':0}, {'x':1,'y':2,'z':0}], 0, True)
        self._add_block(2, [{'x':2,'y':1,'z':0}, {'x':3,'y':1,'z':0}, {'x':3,'y':2,'z':0}], 0, True)
        self.learner = random.choice([1, 2])
        self.opponent = 2 if self.learner == 1 else 1
        self.current_player = 1
        if self.opponent == 1: self._smart_bot_turn()
        self.prev_score = self._evaluate_board(self.learner)
        return self._get_obs(), {}

    def step(self, action):
        if not self._execute_move(self.learner, action): return self._get_obs(), -5.0, True, False, {}
        if self._check_win() == self.learner: return self._get_obs(), 100.0, True, False, {}
        current_score = self._evaluate_board(self.learner)
        shaping_reward = (current_score - self.prev_score) * 0.1
        
        self._next_turn()
        self._smart_bot_turn()
        if self._check_win() == self.opponent: return self._get_obs(), -100.0, True, False, {}
        self._next_turn()
        self.prev_score = self._evaluate_board(self.learner)
        terminated = False if self.turn_count <= 100 else True
        return self._get_obs(), shaping_reward - 0.01, terminated, False, {}

    def _next_turn(self):
        self.turn_count += 1
        self.current_player = 2 if self.current_player == 1 else 1
        if self.blocks_left[self.current_player] > 0: self.phase = 'PLACEMENT'
        else: self.phase = 'MOVEMENT'

    def _evaluate_board(self, player):
        top_map = np.zeros((5,5), dtype=int)
        for y in range(5):
            for x in range(5):
                for z in range(4, -1, -1):
                    if self.board[z][y][x] != 0: top_map[y][x] = self.board[z][y][x]; break
        score = 0; opp = 3 - player; lines = []
        for y in range(5): lines.append(top_map[y, :])
        for x in range(5): lines.append(top_map[:, x])
        lines.append(np.diag(top_map)); lines.append(np.diag(np.fliplr(top_map)))
        for line in lines:
            l = line.tolist()
            score += self._count_pattern(l, player, 4)*20 + self._count_pattern(l, player, 3)*10
            score -= self._count_pattern(l, opp, 4)*25 + self._count_pattern(l, opp, 3)*15
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

    def _smart_bot_turn(self):
        mask = self._get_action_masks_for_player(self.opponent)
        legal_moves = np.where(mask)[0].tolist()
        if not legal_moves: return
        for action in legal_moves:
            if self._simulate_win(self.opponent, action): self._execute_move(self.opponent, action); return
        best_action = -1; max_eval = -99999
        candidates = random.sample(legal_moves, min(len(legal_moves), 15))
        for action in candidates:
            backup_board = self.board.copy(); backup_blocks = [b.copy() for b in self.blocks]
            backup_left = self.blocks_left.copy(); backup_phase = self.phase
            if self._execute_move(self.opponent, action):
                eval_score = self._evaluate_board(self.opponent)
                if eval_score > max_eval: max_eval = eval_score; best_action = action
            self.board = backup_board; self.blocks = backup_blocks
            self.blocks_left = backup_left; self.phase = backup_phase
        final_action = best_action if best_action != -1 else random.choice(legal_moves)
        self._execute_move(self.opponent, final_action)

    def _execute_move(self, player, action):
        is_placement = (self.blocks_left[player] > 0)
        if is_placement:
            cells, sh = self._get_landing_cells(action, player)
            if cells: self._add_block(player, cells, sh); self.blocks_left[player] -= 1; return True
            return False
        else:
            my_blocks = [b for b in self.blocks if b['player'] == player and not b.get('fixed')]
            target_sh = action % 8
            for b in my_blocks:
                if not self._can_pick(b): continue
                orig_cells = b['cells']
                for c in orig_cells: self.board[c['z']][c['y']][c['x']] = 0
                landing_cells, _ = self._get_landing_cells(action, player, ignore_block_cells=orig_cells)
                if landing_cells:
                    c_set = set((c['x'],c['y'],c['z']) for c in landing_cells)
                    o_set = set((c['x'],c['y'],c['z']) for c in orig_cells)
                    if c_set != o_set: self._remove_block_internal(b['id']); self._add_block(player, landing_cells, target_sh, block_id=b['id']); return True
                for c in orig_cells: self.board[c['z']][c['y']][c['x']] = player
            return False

    def _get_landing_cells(self, action_idx, player, ignore_block_cells=None):
        sh, px, py = action_idx % 8, (action_idx // 8) % 5, (action_idx // 8) // 5
        base_shape = self.SHAPES[sh]
        for dz in range(5):
            test_cells = [{'x': px + dx, 'y': py + dy, 'z': dz + d_z_offset} for dx, dy, d_z_offset in base_shape]
            if any(c['z'] > 4 for c in test_cells): break
            if self._check_validity_strict(player, test_cells, ignore_block_cells): return test_cells, sh
        return None, None

    def _check_validity_strict(self, player, cells, ignore_block_cells=None):
        ignore_set = set()
        if ignore_block_cells:
            for c in ignore_block_cells: ignore_set.add((c['x'], c['y'], c['z']))
        for c in cells:
            if not (0<=c['x']<5 and 0<=c['y']<5 and 0<=c['z']<5): return False
            if self.board[c['z']][c['y']][c['x']] != 0:
                if (c['x'], c['y'], c['z']) not in ignore_set: return False
        if ignore_block_cells and set((c['x'], c['y'], c['z']) for c in cells) == ignore_set: return False
        ground_contact = sum(1 for c in cells if c['z'] == 0)
        for c in cells:
            if c['z'] > 0:
                bx, by, bz = c['x'], c['y'], c['z']-1
                has_support = (self.board[bz][by][bx] != 0)
                if (bx, by, bz) in ignore_set: has_support = False
                is_self = any(sc['x']==bx and sc['y']==by and sc['z']==bz for sc in cells)
                if not has_support and not is_self: return False
        if ground_contact > 0 and ground_contact not in [1, 3]: return False
        if not ignore_block_cells and self.turn_count < 2:
            restr = ["0,3", "0,4", "1,4", "3,0", "4,0", "4,1"]
            for c in cells:
                if c['z']==0 and f"{c['x']},{c['y']}" in restr: return False
        return True

    def _can_pick(self, block):
        if block.get('fixed'): return False
        for c in block['cells']:
            if c['z'] >= 4: continue
            if self.board[c['z']+1][c['y']][c['x']] != 0:
                is_self = any(sc['x']==c['x'] and sc['y']==c['y'] and sc['z']==c['z']+1 for sc in block['cells'])
                if not is_self: return False
        return True

    def _add_block(self, player, cells, shape_idx, is_fixed=False, block_id=None):
        if block_id is None: block_id = self.turn_count * 10000 + len(self.blocks)
        self.blocks.append({'id': block_id, 'player': player, 'cells': cells, 'shapeIdx': shape_idx, 'fixed': is_fixed})
        for c in cells: self.board[c['z']][c['y']][c['x']] = player

    def _remove_block_internal(self, block_id):
        idx = next((i for i, b in enumerate(self.blocks) if b['id'] == block_id), -1)
        if idx != -1: self.blocks.pop(idx)

    def _simulate_win(self, player, action):
        backup_board = self.board.copy(); backup_blocks = [b.copy() for b in self.blocks]
        backup_left = self.blocks_left.copy(); backup_phase = self.phase
        win = False
        if self._execute_move(player, action):
            if self._check_win() == player: win = True
        self.board = backup_board; self.blocks = backup_blocks
        self.blocks_left = backup_left; self.phase = backup_phase
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
        phase_val = 1.0 if self.blocks_left[self.learner] > 0 else 0.0
        state_channel = np.full(self.board_shape, phase_val, dtype=np.float32)
        return np.stack([my_stones, opp_stones, state_channel], axis=0)
    
    def action_masks(self): return self._get_action_masks_for_player(self.learner)
    def _get_action_masks_for_player(self, player):
        mask = np.zeros(200, dtype=bool)
        is_placement = (self.blocks_left[player] > 0)
        if is_placement:
            for i in range(200):
                cells, _ = self._get_landing_cells(i, player)
                if cells: mask[i] = True
        else:
            my_blocks = [b for b in self.blocks if b['player'] == player and not b.get('fixed') and self._can_pick(b)]
            for i in range(200):
                possible = False
                for b in my_blocks:
                    for c in b['cells']: self.board[c['z']][c['y']][c['x']] = 0
                    l_cells, _ = self._get_landing_cells(i, player, ignore_block_cells=b['cells'])
                    for c in b['cells']: self.board[c['z']][c['y']][c['x']] = player
                    if l_cells:
                         c_set = set((c['x'],c['y'],c['z']) for c in l_cells)
                         o_set = set((c['x'],c['y'],c['z']) for c in b['cells'])
                         if c_set != o_set: possible = True; break
                if possible: mask[i] = True
        return mask

# ============================================================================
# 💾 저장 콜백 및 실행
# ============================================================================
class LocalSaveCallback(BaseCallback):
    def __init__(self, save_freq=5000, save_path="./flight_models", verbose=1):
        super(LocalSaveCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.gen_count = 10 # Gen 10부터 이어한다고 가정 (원하는 숫자로 변경 가능)
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.gen_count += 1
            save_name = os.path.join(self.save_path, f"flight_gen_{self.gen_count}")
            self.model.save(save_name)
            if self.verbose > 0:
                print(f"✈️ [비행 중] 모델 저장됨: {save_name} (Step: {self.num_timesteps})")
        return True

def mask_fn(env): return env.get_wrapper_attr('action_masks')()

if __name__ == '__main__':
    vec_env = DummyVecEnv([lambda: ActionMasker(SpartaOmokEnv(), mask_fn) for _ in range(N_ENVS)])
    policy_kwargs = dict(features_extractor_class=Omok3D_CNN, features_extractor_kwargs=dict(features_dim=256), net_arch=[])
    
    # 모델 로드 (커스텀 객체 필수)
    custom_objects = {
        "policy_kwargs": policy_kwargs,
        "learning_rate": 0.0003, # 저장된 모델의 설정을 덮어씌울 때 필요할 수 있음
        "clip_range": 0.2
    }
    
    try:
        print("⚡ 모델 로딩 중...")
        model = MaskablePPO.load(model_path, env=vec_env, device=DEVICE, custom_objects=custom_objects)
        print(f"✅ '{LOAD_MODEL_NAME}' 로드 성공! 비행기 모드 학습을 시작합니다.")
    except Exception as e:
        print(f"❌ 로드 실패 오류: {e}")
        print("혹시 stable-baselines3 버전이 다른가요? 그래도 일단 새 모델로 시작합니다.")
        model = MaskablePPO("CnnPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=1024, batch_size=64, gamma=0.99, device=DEVICE, policy_kwargs=policy_kwargs)

    print(f"🔥 학습 시작! (목표: 무한, 멈추려면 Ctrl+C)")
    try:
        # reset_num_timesteps=False: 이전 학습 그래프(엔트로피 등)를 이어서 감
        model.learn(total_timesteps=10_000_000, callback=LocalSaveCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR), reset_num_timesteps=False)
        model.save("flight_final_completed")
        print("✅ 목적지 도착 (학습 완료)")
    except KeyboardInterrupt:
        print("\n🛑 비행 중단 요청 감지! (Ctrl+C)")
        model.save("flight_interrupted")
        print("✅ 안전하게 저장했습니다. (flight_interrupted.zip)")
        time.sleep(2)