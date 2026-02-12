# export_onnx.py
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO

# 1. AI 뇌 구조 정의 (학습 때랑 똑같이)
class Omok3D_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
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

# 2. 모델 로드
model_path = "sparta_cnn_final.zip"  # <--- 학습된 파일 이름 확인!
custom_objects = {
    "policy_kwargs": {
        "features_extractor_class": Omok3D_CNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": []
    }
}

print("📂 모델 로드 중...")
model = MaskablePPO.load(model_path, device='cpu', custom_objects=custom_objects)

# 3. ONNX로 추출 (핵심)
class OnnxablePolicy(nn.Module):
    def __init__(self, extractor, action_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
    
    def forward(self, observation):
        # 1. 3D CNN으로 특징 추출
        features = self.extractor(observation)
        # 2. 행동 결정 (Logits 반환)
        action_logits = self.action_net(features)
        return action_logits

onnx_policy = OnnxablePolicy(model.policy.features_extractor, model.policy.action_net)
onnx_policy.eval()

# 가짜 입력 데이터 (형식 맞추기용: 배치1, 채널2, 5x5x5)
dummy_input = torch.randn(1, 2, 5, 5, 5)

print("⚡ ONNX 변환 중...")
torch.onnx.export(
    onnx_policy,
    dummy_input,
    "omok_model.onnx",  # 저장될 파일 이름
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print("🎉 변환 완료! 'omok_model.onnx' 파일이 생성되었습니다.")
print("이제 이 파일을 index.html 옆에 두세요.")