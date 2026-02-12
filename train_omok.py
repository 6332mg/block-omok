import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ============================================================================
# ⚡ [Speed Up] 연산 최적화된 스파르타 환경 (로컬 PC용)
# ============================================================================
class SpartaOmokEnv(gym.Env):
    # ⚡ [MCTS 탑재] 스마트 봇 (이제 수읽기를 합니다!)
    def _smart_bot_turn(self):
        legal_moves = self._get_legal_moves_indices(self.opponent)
        if not legal_moves: return

        # 1. 킬각 (계산 0초컷이므로 유지)
        for action in legal_moves:
            if self._simulate_move_fast(self.opponent, action):
                self._execute_move(self.opponent, action)
                return

        # 2. 방어 (계산 0초컷이므로 유지)
        opp_moves = self._get_legal_moves_indices(self.learner)
        threats = []
        for action in opp_moves:
             if self._simulate_move_fast(self.learner, action):
                threats.append(action)
        for threat in threats:
            if threat in legal_moves:
                self._execute_move(self.opponent, threat)
                return

        # 3. 🔥 MCTS (몬테카를로 탐색)
        # 여기서 시간을 씁니다. n_simulations가 높을수록 똑똑하지만 느려집니다.
        # 로컬 PC(i5) 성능을 고려해 30번만 수읽기 합니다. (웹사이트는 1500번 함)
        best_action = self._run_mcts_simulation(legal_moves, simulations_per_move=3, max_depth=5)
        self._execute_move(self.opponent, best_action)

    # 🧠 MCTS 시뮬레이션 엔진
    def _run_mcts_simulation(self, candidates, simulations_per_move=3, max_depth=5):
        best_score = -9999
        best_move = random.choice(candidates) # 기본값

        # 모든 후보 수에 대해 가상으로 둬봅니다.
        for move in candidates:
            wins = 0
            
            # 각 후보마다 N번씩 랜덤 게임을 끝까지(혹은 depth까지) 둬봅니다.
            for _ in range(simulations_per_move):
                # 1. 가상 보드 복사 (여기가 속도 병목 구간)
                temp_board = self.board.copy()
                
                # 2. 첫 수 두기
                sh, px, py = move%8, (move//8)%5, (move//8)//5
                cells = self._get_cells(px, py, sh)
                for c in cells: temp_board[c['z']][c['y']][c['x']] = self.opponent
                
                # 3. 랜덤 시뮬레이션 시작 (Rollout)
                sim_turn = 0
                current_sim_player = self.learner # 다음 턴은 상대방
                my_sim_id = self.opponent
                
                while sim_turn < max_depth:
                    # 승리 체크 (간단 버전) - 속도를 위해 정밀 체크 생략 가능하면 생략
                    # 하지만 여기선 정확도를 위해 체크합니다.
                    if self._check_win_simulation(temp_board) == my_sim_id:
                        wins += 1
                        break
                    
                    # 랜덤으로 아무거나 둠 (가상 상대방)
                    # (정석 구현은 legal move를 다 찾아야 하지만 너무 느리므로 완전 랜덤 좌표 찍기)
                    # 속도 최적화를 위해 '빈 공간 찾기' 대신 그냥 턴만 넘기는 식으로 depth만 체크할 수도 있음
                    # 여기서는 '약식'으로 빈 공간 아무데나 하나 채우는 걸로 가정
                    empty_spots = np.argwhere(temp_board == 0)
                    if len(empty_spots) == 0: break
                    
                    # 랜덤 착수
                    choice = empty_spots[random.randint(0, len(empty_spots)-1)]
                    temp_board[choice[0]][choice[1]][choice[2]] = current_sim_player
                    
                    # 턴 교체
                    current_sim_player = my_sim_id if current_sim_player != my_sim_id else (3 - my_sim_id)
                    sim_turn += 1
            
            # 승률 계산
            if wins > best_score:
                best_score = wins
                best_move = move
        
        return best_move

    # 시뮬레이션용 승리 체크 (기존 함수 재활용을 위해 self.board 대신 인자 받음)
    def _check_win_simulation(self, board_arr):
        # 기존 _check_win 로직을 board_arr 대상으로 수행하도록 복사하거나 수정 필요
        # 편의상 기존 로직을 복사해서 board_arr만 쓰도록 함 (속도상 이 방법이 최선)
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
    def __init__(self):
        super(SpartaOmokEnv, self).__init__()
        self.board_shape = (5, 5, 5)
        self.action_space = spaces.Discrete(200)
        self.observation_space = spaces.Box(low=0, high=1, shape=(250,), dtype=np.int8)

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

        if self.opponent == 1:
            self._smart_bot_turn()

        return self._get_obs(), {}

    def _get_obs(self):
        flat_board = self.board.flatten()
        my_stones = (flat_board == self.learner).astype(np.int8)
        opp_stones = (flat_board == self.opponent).astype(np.int8)
        return np.concatenate([my_stones, opp_stones])

    def step(self, action):
        if not self._execute_move(self.learner, action):
            return self._get_obs(), -50, True, False, {}

        if self._check_win() == self.learner:
            return self._get_obs(), 100, True, False, {}

        self._next_turn()

        # 봇 착수 (최적화됨)
        self._smart_bot_turn()

        if self._check_win() == self.opponent:
            return self._get_obs(), -500, True, False, {}

        self._next_turn()

        terminated = False
        if self.turn_count > 100: terminated = True
        return self._get_obs(), 0.1, terminated, False, {}

    
    # ⚡ [핵심] 초고속 시뮬레이션 (Copy 없음)
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

# ============================================================================
# 💾 [Local Save] 저장 콜백 (로컬 PC용)
# ============================================================================
class LocalSaveCallback(BaseCallback):
    def __init__(self, save_freq=100000, save_path="./models", verbose=0):
        super(LocalSaveCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.gen_count = 0
        os.makedirs(self.save_path, exist_ok=True) # 폴더 없으면 생성

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.gen_count += 1
            path = os.path.join(self.save_path, f"sparta_gen_{self.gen_count}")
            self.model.save(path)
            if self.verbose > 0:
                print(f"💾 [Local] Generation {self.gen_count} Saved! (Step: {self.num_timesteps}) at {path}")
        return True

def mask_fn(env): return env.get_wrapper_attr('action_masks')()

# ============================================================================
# 🏃‍♂️ [Main] 실행부
# ============================================================================
# ============================================================================
# 🏃‍♂️ [Main] 실행부 (수정됨: 이어하기 기능 추가)
# ============================================================================
if __name__ == '__main__':
    # i5-1135G7은 4코어 8스레드이므로 n_envs=4 권장
    n_envs = 4 
    
    # GPU 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 하드웨어 가속 확인: {device.upper()} 모드로 실행합니다.")

    # 환경 생성
    vec_env = SubprocVecEnv([lambda: ActionMasker(SpartaOmokEnv(), mask_fn) for _ in range(n_envs)])

    # 🌟 [핵심] 저장된 모델이 있으면 불러오고, 없으면 새로 만들기
    load_filename = "my_model.zip"  # 폴더에 넣어둔 파일 이름
    
    if os.path.exists(load_filename):
        print(f"♻️ 발견! '{load_filename}' 모델을 로드하여 훈련을 이어갑니다...")
        # custom_objects는 훈련 환경 버전에 따라 필요할 수 있음 (일단 기본 로드)
        model = MaskablePPO.load(load_filename, env=vec_env, device=device)
        
        # 학습률(learning_rate) 등 일부 설정은 새로 덮어쓰기 위해 다시 설정
        model.learning_rate = 0.0003
        model.n_steps = 2048
        model.batch_size = 256
        model.gamma = 0.99
    else:
        print("✨ 저장된 모델이 없습니다. 0부터 새로운 훈련을 시작합니다!")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            device=device,
            policy_kwargs=dict(net_arch=[1024, 1024])
        )

    print("🔥 [Local PC Mode] 로컬 스파르타 훈련 시작! 🔥")
    
    # 500만 번 추가 훈련
    total_steps = 5000000
    callback = LocalSaveCallback(save_freq=100000, save_path="./models", verbose=1)

    try:
        model.learn(total_timesteps=total_steps, callback=callback)
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단됨. 현재 상태를 저장합니다...")
        model.save("models/interrupted_model")
        print("✅ 저장 완료.")

    print("✅ 지옥 훈련 완료!")
    
    # JSON 추출 및 저장
    import json
    params = {}
    p_net = model.policy.mlp_extractor.policy_net
    a_net = model.policy.action_net
    
    params['fc0_w'] = p_net[0].weight.detach().cpu().numpy().tolist()
    params['fc0_b'] = p_net[0].bias.detach().cpu().numpy().tolist()
    params['fc1_w'] = p_net[2].weight.detach().cpu().numpy().tolist()
    params['fc1_b'] = p_net[2].bias.detach().cpu().numpy().tolist()
    params['fc2_w'] = a_net.weight.detach().cpu().numpy().tolist()
    params['fc2_b'] = a_net.bias.detach().cpu().numpy().tolist()

    with open("legendary_ai_local_final.json", "w") as f:
        json.dump(params, f)
    print("🎉 최종 JSON 파일 생성 완료: legendary_ai_local_final.json")