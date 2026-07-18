"""DQN으로 CartPole-v1을 학습하는 스크립트.

실행하면 학습 후 saved_models/ 에 가중치를, training_logs/ 에 로그와 그래프를 저장합니다.
저장된 가중치는 evaluate.py 로 불러와 성능을 확인할 수 있습니다.
"""

import random
from collections import deque, namedtuple
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 하나의 경험(transition)을 저장하기 위한 자료형입니다.
# state: 현재 상태 / action: 선택한 행동 / reward: 받은 보상
# next_state: 다음 상태 / done: 에피소드 종료 여부
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    # 과거 경험을 저장하고 무작위로 꺼내기 위한 버퍼입니다.
    # DQN은 연속된 샘플 대신 랜덤 샘플을 사용해 샘플 간 상관관계를 줄이고
    # 학습을 더 안정적으로 만듭니다.
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    # 상태를 입력받아 각 행동의 Q값을 출력하는 신경망입니다.
    # CartPole은 상태가 4차원, 행동이 2개이므로 입력 4 → 출력 2가 됩니다.
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


def save_model(model, save_path: Path):
    # 학습이 끝난 정책 네트워크의 가중치를 파일로 저장합니다.
    # evaluate.py 에서 이 가중치를 다시 불러와 사용합니다.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"모델 가중치를 저장했습니다: {save_path}")


def choose_action(state, policy_net, epsilon, action_dim, device):
    # epsilon-greedy 전략입니다.
    # 확률 epsilon으로는 무작위 행동(탐험), 그 외에는 Q값이 가장 큰 행동(활용)을 선택합니다.
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    # 버퍼에 배치 크기만큼 데이터가 쌓이기 전에는 학습하지 않습니다.
    if len(memory) < batch_size:
        return None

    # [(s1, a1, ...), (s2, a2, ...)] 형태의 샘플 목록을
    # (s1, s2, ...), (a1, a2, ...) 처럼 항목별로 묶어 배치 텐서로 변환합니다.
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # 예측값 Q(s, a): 네트워크가 출력한 Q값들 중,
    # 실제로 선택했던 행동 위치의 값만 gather로 골라냅니다. (batch, 2) -> (batch, 1)
    current_q = policy_net(states).gather(1, actions)

    # 타깃값 r + gamma * max_a' Q_target(s', a'):
    # 타깃 네트워크로 다음 상태의 최대 Q값을 계산합니다.
    # done이면 다음 상태가 없으므로 (1 - dones)를 곱해 보상만 남깁니다.
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        target_q = rewards + gamma * max_next_q * (1 - dones)

    # DQN의 손실은 예측 Q값과 타깃 Q값의 차이(MSE)입니다.
    loss = nn.MSELoss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def moving_average(values, window_size: int):
    # 최근 성능 흐름을 보기 쉽도록 이동평균을 계산합니다.
    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window_size + 1)
        averaged.append(float(np.mean(values[start : idx + 1])))
    return averaged


def save_training_log(log_dir: Path, scores, avg_scores, losses, epsilons):
    # 학습이 끝난 뒤 다시 분석할 수 있도록 로그를 CSV 파일로 저장합니다.
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"

    with csv_path.open("w", encoding="utf-8") as file:
        file.write("episode,score,moving_avg_20,loss,epsilon\n")
        for episode, (score, avg_score, loss, epsilon) in enumerate(
            zip(scores, avg_scores, losses, epsilons), start=1
        ):
            file.write(f"{episode},{score},{avg_score},{loss},{epsilon}\n")

    print(f"학습 로그를 저장했습니다: {csv_path}")


def plot_training_log(log_dir: Path, scores, avg_scores, losses):
    # 학습 종료 후 점수와 손실 변화를 그래프로 저장하고 화면에 띄웁니다.
    episodes = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, scores, label="Score per episode", alpha=0.6)
    plt.plot(episodes, avg_scores, label="Moving average (20)", linewidth=2)
    plt.title("DQN CartPole Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, losses, label="Loss", color="tomato")
    plt.title("DQN CartPole Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    image_path = log_dir / "training_plot.png"
    plt.savefig(image_path, dpi=150)
    print(f"학습 그래프를 저장했습니다: {image_path}")

    plt.show()


def train():
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # policy_net: 매 스텝 학습되는 네트워크
    # target_net: 타깃 Q값 계산용으로, 주기적으로만 policy_net을 복사해 학습을 안정화합니다.
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayBuffer(capacity=10000)

    num_episodes = 300
    batch_size = 64
    gamma = 0.99  # 미래 보상 할인율

    epsilon = 1.0  # 탐험 초기 확률
    epsilon_min = 0.05  # 탐험 확률의 하한
    epsilon_decay = 0.995  # 매 에피소드 후 epsilon에 곱하는 감소율

    target_update_interval = 10  # 10 에피소드마다 target_net에 policy_net 가중치를 복사

    scores = []
    avg_scores = []
    losses = []
    epsilons = []
    log_dir = Path("training_logs")
    model_dir = Path("saved_models")
    model_path = model_dir / "dqn_cartpole_policy.pth"

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []

        # CartPole-v1은 최대 500스텝에서 에피소드가 종료됩니다.
        for step in range(1, 501):
            action = choose_action(state, policy_net, epsilon, action_dim, device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 환경 진행은 스텝마다 순차적으로 이루어지고,
            # 학습은 버퍼에서 랜덤하게 뽑은 과거 경험 배치로 이루어집니다.
            memory.push(state, action, reward, next_state, done)
            loss = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                memory=memory,
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        # epsilon을 조금씩 줄여 탐험에서 활용 위주로 전환합니다.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 타깃 네트워크를 주기적으로 갱신해 학습을 안정화합니다.
        if episode % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        scores.append(total_reward)
        avg_score = np.mean(scores[-20:])
        avg_scores.append(float(avg_score))
        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        losses.append(mean_loss)
        epsilons.append(float(epsilon))

        print(
            f"Episode {episode:3d} | "
            f"Score: {total_reward:3.0f} | "
            f"Avg(20): {avg_score:6.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Loss: {mean_loss:.4f}"
        )

        # 최근 20판 평균이 충분히 높으면 학습 성공으로 보고 조기 종료합니다.
        if avg_score >= 475:
            print("학습이 충분히 진행되어 조기 종료합니다.")
            break

    env.close()
    save_model(policy_net, model_path)
    save_training_log(log_dir, scores, avg_scores, losses, epsilons)
    plot_training_log(log_dir, scores, avg_scores, losses)


if __name__ == "__main__":
    train()
