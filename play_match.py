import os
import time
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ============================================================================
# 🧬 1. 학습할 때 썼던 3D CNN 구조 (뇌 구조 복원)
# ============================================================================
class Omok3D_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(Omok3D_CNN, self).__init__(observation_space, features_dim)
        # 학습 코드와 완전히 동일한 구조여야 함
        self.cnn = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
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
# 🏟️ 2. 게임 환경 (학습 코드와 동일)
# ============================================================================
class SpartaOmokEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 5, 5, 5), dtype=np.float32)
        self.action_space = spaces.Discrete(200)
        self.SHAPES = [
            [(0,0,0), (1,0,0), (0,1,0)], [(0,0,0), (1,0,0), (0,-1,0)],
            [(0,0,0), (-1,0,0), (0,-1,0)], [(0,0,0), (-1,0,0), (0,1,0)],
            [(0,0,0), (0,0,1), (1,0,1)], [(0,0,0), (0,0,1), (-1,0,1)],
            [(0,0,0), (0,0,1), (0,1,1)], [(0,0,0), (0,0,1), (0,-1,1)]
        ]
        self.reset()

    def reset(self, seed=None, options=None):
        self.board = np.zeros((5,5,5), dtype=np.int8)
        self.blocks = []
        self.turn_count = 0
        self.blocks_left = {1: 4, 2: 4}
        self.phase = 'PLACEMENT'
        # 초기 배치 (고정)
        self._add_block(1, [{'x':1,'y':3,'z':0}, {'x':2,'y':3,'z':0}, {'x':1,'y':2,'z':0}], 0, True)
        self._add_block(2, [{'x':2,'y':1,'z':0}, {'x':3,'y':1,'z':0}, {'x':3,'y':2,'z':0}], 0, True)
        self.learner = 1 
        self.current_player = 1
        return self._get_obs(), {}

    def _get_obs(self):
        my_stones = (self.board == self.learner).astype(np.float32)
        opp_stones = (self.board == (3-self.learner)).astype(np.float32)
        return np.stack([my_stones, opp_stones], axis=0)

    def step(self, action): pass 

    # 유틸리티 함수들
    def _add_block(self, player, cells, shape_idx, is_fixed=False, block_id=None):
        if block_id is None: block_id = self.turn_count * 10000 + len(self.blocks)
        self.blocks.append({'id': block_id, 'player': player, 'cells': cells, 'shapeIdx': shape_idx, 'fixed': is_fixed})
        for c in cells: self.board[c['z']][c['y']][c['x']] = player

    def _remove_block(self, block_id):
        idx = next((i for i, b in enumerate(self.blocks) if b['id'] == block_id), -1)
        if idx != -1:
            block = self.blocks.pop(idx)
            for c in block['cells']: self.board[c['z']][c['y']][c['x']] = 0
            return block
        return None

    def _get_cells(self, bx, by, shape_idx):
        return [{'x': bx + dx, 'y': by + dy, 'z': dz} for dx, dy, dz in self.SHAPES[shape_idx]]

    def _can_pick(self, block):
        if block.get('fixed'): return False
        for c in block['cells']:
            if c['z'] >= 4: continue
            if self.board[c['z']+1][c['y']][c['x']] != 0:
                is_self = any(sc['x']==c['x'] and sc['y']==c['y'] and sc['z']==c['z']+1 for sc in block['cells'])
                if not is_self: return False
        return True

    def check_validity(self, player, cells, is_movement=False, original_cells=None):
        for c in cells:
            if not (0<=c['x']<5 and 0<=c['y']<5 and 0<=c['z']<5): return False, "범위 초과"
            if self.board[c['z']][c['y']][c['x']] != 0: return False, "이미 돌이 있음"
        if is_movement and original_cells:
            c_set = set((c['x'],c['y'],c['z']) for c in cells)
            o_set = set((c['x'],c['y'],c['z']) for c in original_cells)
            if c_set == o_set: return False, "제자리 착수 불가"
        ground = sum(1 for c in cells if c['z']==0)
        if ground != 3 and ground != 1: return False, "바닥 닿는 면적 위반 (1 or 3)"
        for c in cells:
            if c['z'] > 0:
                has_sup = self.board[c['z']-1][c['y']][c['x']] != 0
                is_self = any(sc['x']==c['x'] and sc['y']==c['y'] and sc['z']==c['z']-1 for sc in cells)
                if not has_sup and not is_self: return False, "공중 부양 불가"
        if not is_movement and self.turn_count < 2:
            restricted = ["0,3", "0,4", "1,4", "3,0", "4,0", "4,1"]
            for c in cells:
                if c['z']==0 and f"{c['x']},{c['y']}" in restricted: return False, "초반 금지구역"
        return True, "OK"

    def check_win(self):
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

    def action_masks(self):
        mask = np.zeros(200, dtype=bool)
        player = self.current_player
        if self.phase == 'PLACEMENT':
            if self.blocks_left[player] > 0:
                for i in range(200):
                    sh, px, py = i%8, (i//8)%5, (i//8)//5
                    cells = self._get_cells(px, py, sh)
                    if self.check_validity(player, cells)[0]: mask[i] = True
        else: # MOVEMENT
            my_blocks = [b for b in self.blocks if b['player'] == player and not b.get('fixed') and self._can_pick(b)]
            for b in my_blocks:
                orig = b['cells']
                for c in orig: self.board[c['z']][c['y']][c['x']] = 0
                for i in range(200):
                    if mask[i]: continue
                    sh, px, py = i%8, (i//8)%5, (i//8)//5
                    cells = self._get_cells(px, py, sh)
                    if self.check_validity(player, cells, True, orig)[0]: mask[i] = True
                for c in orig: self.board[c['z']][c['y']][c['x']] = player
        return mask

def mask_fn(env): return env.action_masks()

# ============================================================================
# ⚔️ 3. 인간 VS AI 대결 로직 (수정됨: 불러오기 옵션 추가)
# ============================================================================
def play_match():
    MODEL_PATH = "sparta_cnn_final.zip" 
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ {MODEL_PATH} 파일이 없습니다. 경로를 확인하세요!")
        return

    env = SpartaOmokEnv()
    
    # 🌟 [핵심 수정] 학습할 때와 똑같은 옵션을 줘야 에러가 안 남!
    # net_arch=[] : 중간 레이어 없이 바로 연결한다는 뜻
    custom_objects = {
        "policy_kwargs": {
            "features_extractor_class": Omok3D_CNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [] 
        }
    }

    try:
        model = MaskablePPO.load(MODEL_PATH, env=env, custom_objects=custom_objects)
        print("🤖 [SYSTEM] AI 모델 로드 성공! (특이점 버전)")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    obs, _ = env.reset()
    
    # 선공 결정
    print("\n🎲 동전을 던집니다...")
    if np.random.rand() > 0.5:
        print("🧑 당신이 선공(흑돌/Player 1)입니다!")
        ai_player = 2
    else:
        print("🤖 AI가 선공(흑돌/Player 1)입니다!")
        ai_player = 1
    
    while True:
        # 보드 출력
        print(f"\n[Turn {env.turn_count}] 현재 차례: {'🤖 AI' if env.current_player == ai_player else '🧑 인간'}")
        print(f"단계: {env.phase}, 남은 블록(흑:{env.blocks_left[1]}, 백:{env.blocks_left[2]})")
        
        for z in range(4, -1, -1):
            print(f"--- {z+1}층 ---")
            for y in range(4, -1, -1):
                line = ""
                for x in range(5):
                    v = env.board[z][y][x]
                    if v == 0: line += ". "
                    elif v == 1: line += "● "
                    elif v == 2: line += "○ "
                print(line)
        
        winner = env.check_win()
        if winner != 0:
            print(f"\n🎉 승리: {'🤖 AI' if winner == ai_player else '🧑 인간'}")
            break
        
        if env.current_player == ai_player:
            print("🤖 AI가 생각 중...", end="", flush=True)
            time.sleep(0.5)
            
            action, _ = model.predict(obs, action_masks=mask_fn(env), deterministic=True)
            
            sh, px, py = int(action)%8, (int(action)//8)%5, (int(action)//8)//5
            print(f" -> 착수 완료! (위치: {px},{py} / 모양: {sh})")
            
            if env.phase == 'PLACEMENT':
                cells = env._get_cells(px, py, sh)
                env._add_block(ai_player, cells, sh)
                env.blocks_left[ai_player] -= 1
            else:
                mask = env.action_masks()
                if not mask[action]: 
                    print("❌ AI가 불가능한 수를 뒀습니다.")
                    break
                
                my_blocks = [b for b in env.blocks if b['player'] == ai_player and not b.get('fixed') and env._can_pick(b)]
                moved = False
                cells = env._get_cells(px, py, sh)
                for b in my_blocks:
                    orig = b['cells']
                    for c in orig: env.board[c['z']][c['y']][c['x']] = 0
                    if env.check_validity(ai_player, cells, True, orig)[0]:
                        for c in orig: env.board[c['z']][c['y']][c['x']] = ai_player 
                        env._remove_block(b['id'])
                        env._add_block(ai_player, cells, sh)
                        moved = True
                        print(f"🤖 AI가 블록을 이동시켰습니다.")
                        break
                    for c in orig: env.board[c['z']][c['y']][c['x']] = ai_player
                if not moved: print("❌ AI 이동 오류")

        else:
            while True:
                try:
                    if env.phase == 'PLACEMENT':
                        user_in = input("착수 입력 (x y 모양[0-7]): ").split()
                        if len(user_in) != 3: raise ValueError
                        px, py, sh = map(int, user_in)
                        cells = env._get_cells(px, py, sh)
                        ok, msg = env.check_validity(env.current_player, cells)
                        if ok:
                            env._add_block(env.current_player, cells, sh)
                            env.blocks_left[env.current_player] -= 1
                            break
                        else:
                            print(f"🚫 불가: {msg}")
                    else:
                        print("이동 단계입니다. (테스트용이므로 여기서 종료합니다.)")
                        return
                except:
                    print("잘못된 입력. 예: 2 2 0")

        env.turn_count += 1
        env.current_player = 3 - env.current_player
        if env.blocks_left[1] == 0 and env.blocks_left[2] == 0: env.phase = 'MOVEMENT'
        elif env.blocks_left[env.current_player] == 0 and env.phase == 'PLACEMENT': env.phase = 'MOVEMENT'
        
        if env.current_player == ai_player:
            env.learner = ai_player 
            obs = env._get_obs()

if __name__ == '__main__':
    play_match()