"""train.py 로 학습한 DQN 모델을 불러와 CartPole-v1에서 성능을 확인하는 스크립트.

렌더링 창이 열리며 5개 에피소드를 실행하고 점수를 출력합니다.
"""

from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn


class DQN(nn.Module):
    # 가중치를 그대로 불러와야 하므로 학습 때 사용한 네트워크 구조와 완전히 같아야 합니다.
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def run_inference():
    # 학습이 아니라 저장된 모델의 성능을 눈으로 확인하는 단계이므로
    # render_mode="human"으로 환경을 생성합니다.
    env = gym.make("CartPole-v1", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim).to(device)

    # train.py 에서 저장한 가중치 중, 학습 중 평가 점수가 가장 높았던 시점의
    # best 모델을 우선 사용하고, 없으면 마지막 에피소드의 가중치를 사용합니다.
    best_path = Path("saved_models/dqn_cartpole_best.pth")
    final_path = Path("saved_models/dqn_cartpole_policy.pth")
    model_path = best_path if best_path.exists() else final_path
    print(f"모델을 불러옵니다: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    num_episodes = 5

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 추론 단계에서는 탐험(epsilon) 없이
            # Q값이 가장 큰 행동만 선택합니다.
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        print(f"테스트 Episode {episode} | Score: {total_reward}")

    env.close()


if __name__ == "__main__":
    run_inference()
