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
# state: 현재 상태 / action: 선택한 행동 / reward: 받은 보상 / next_state: 다음 상태
# done: 막대가 쓰러져 "실패"로 끝났는지 여부 (terminated만 저장, 시간 초과 truncated는 제외)
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
    # 정책 네트워크의 가중치를 파일로 저장합니다.
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
    # 실패(terminated)면 다음 상태가 없으므로 (1 - dones)를 곱해 보상만 남깁니다.
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        target_q = rewards + gamma * max_next_q * (1 - dones)

    # 손실은 예측 Q값과 타깃 Q값의 차이입니다.
    # MSE 대신 Huber(SmoothL1)를 쓰면 타깃이 크게 튀는 샘플에서
    # 그래디언트가 폭주하는 것을 막아 학습이 더 안정적입니다.
    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    # 그래디언트 크기를 제한해서 한 번의 업데이트로 네트워크가 망가지는 것을 방지합니다.
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def evaluate_policy(policy_net, device, num_episodes: int = 5):
    # 진짜 평가지표: 탐험(epsilon) 없이 greedy 정책만으로 몇 판을 돌려 평균 점수를 잽니다.
    # 학습 중 점수는 무작위 탐험이 섞여 있어 정책의 실제 실력보다 낮게 나오고,
    # TD loss는 타깃이 계속 움직여서 성능 지표가 될 수 없기 때문입니다.
    eval_env = gym.make("CartPole-v1")
    total = 0.0

    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(policy_net(state_tensor), dim=1).item())
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total += reward

    eval_env.close()
    return total / num_episodes


def moving_average(values, window_size: int):
    # 최근 성능 흐름을 보기 쉽도록 이동평균을 계산합니다.
    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window_size + 1)
        averaged.append(float(np.mean(values[start : idx + 1])))
    return averaged


def save_training_log(log_dir: Path, scores, avg_scores, losses, epsilons, eval_episodes, eval_scores):
    # 학습이 끝난 뒤 다시 분석할 수 있도록 로그를 CSV 파일로 저장합니다.
    # loss는 성능 지표가 아니라 발산 여부를 확인하는 진단용으로만 남겨둡니다.
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"

    with csv_path.open("w", encoding="utf-8") as file:
        file.write("episode,score,moving_avg_20,loss,epsilon\n")
        for episode, (score, avg_score, loss, epsilon) in enumerate(
            zip(scores, avg_scores, losses, epsilons), start=1
        ):
            file.write(f"{episode},{score},{avg_score},{loss},{epsilon}\n")

    print(f"학습 로그를 저장했습니다: {csv_path}")

    # greedy 평가 점수는 별도 파일로 저장합니다.
    eval_path = log_dir / "eval_log.csv"
    with eval_path.open("w", encoding="utf-8") as file:
        file.write("episode,eval_score\n")
        for episode, eval_score in zip(eval_episodes, eval_scores):
            file.write(f"{episode},{eval_score}\n")

    print(f"평가 로그를 저장했습니다: {eval_path}")


def plot_training_log(log_dir: Path, scores, avg_scores, eval_episodes, eval_scores, solved_threshold):
    # 학습 종료 후 학습 점수(탐험 포함)와 평가 점수(greedy)를 나란히 그립니다.
    episodes = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(12, 5))

    # 왼쪽: 학습 중 epsilon-greedy 정책(행동 정책)의 점수. 탐험이 섞여 있어 참고용입니다.
    plt.subplot(1, 2, 1)
    plt.plot(episodes, scores, label="Score per episode", alpha=0.6)
    plt.plot(episodes, avg_scores, label="Moving average (20)", linewidth=2)
    plt.title("Training Score (epsilon-greedy)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 오른쪽: greedy 정책의 평가 점수. 정책의 실제 실력은 이 곡선으로 판단합니다.
    plt.subplot(1, 2, 2)
    plt.plot(eval_episodes, eval_scores, label="Greedy eval (5 ep avg)", color="seagreen", marker="o")
    plt.axhline(y=solved_threshold, color="gray", linestyle="--", alpha=0.7,
                label=f"Solved ({solved_threshold:.0f})")
    plt.title("Evaluation Score (greedy)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.ylim(0, 520)
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

    num_episodes = 800  # 최대 에피소드 수 (epsilon이 하한 0.05에 도달하는 데 약 600 에피소드가 필요)
    batch_size = 64
    gamma = 0.99  # 미래 보상 할인율

    epsilon = 1.0  # 탐험 초기 확률
    epsilon_min = 0.05  # 탐험 확률의 하한
    epsilon_decay = 0.995  # 매 에피소드 후 epsilon에 곱하는 감소율

    target_update_interval = 10  # 10 에피소드마다 target_net에 policy_net 가중치를 복사
    eval_interval = 10  # 10 에피소드마다 greedy 정책으로 평가
    best_eval_score = -float("inf")

    max_steps = 500  # CartPole-v1은 500스텝에서 에피소드가 강제 종료됩니다

    # 조기 종료 기준: 최대 점수(max_steps)의 95% = 475점.
    # CartPole-v1의 공식 "solved" 기준이 475점이라 0.95를 기본값으로 씁니다.
    # 더 느슨하게 하고 싶으면 비율만 낮추면 됩니다 (예: 0.8 → 400점).
    solved_ratio = 0.95
    solved_threshold = max_steps * solved_ratio

    scores = []
    avg_scores = []
    losses = []
    epsilons = []
    eval_episodes = []
    eval_scores = []
    log_dir = Path("training_logs")
    model_dir = Path("saved_models")
    model_path = model_dir / "dqn_cartpole_policy.pth"
    best_model_path = model_dir / "dqn_cartpole_best.pth"

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []

        for step in range(1, max_steps + 1):
            action = choose_action(state, policy_net, epsilon, action_dim, device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 주의: 버퍼에는 done이 아니라 terminated만 저장합니다.
            # truncated(500스텝 시간 초과)는 실패한 게 아니라 시간이 다 된 것뿐이므로,
            # 이때 다음 상태 가치를 0으로 만들면(done 취급) "오래 버틴 상태는 가치가 낮다"는
            # 잘못된 신호를 학습해 후반 성능이 무너지는 원인이 됩니다.
            memory.push(state, action, reward, next_state, terminated)

            # 환경 진행은 스텝마다 순차적으로 이루어지고,
            # 학습은 버퍼에서 랜덤하게 뽑은 과거 경험 배치로 이루어집니다.
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

        # 주기적으로 greedy 정책을 평가합니다. 조기 종료와 best 모델 판단은
        # 탐험이 섞인 학습 점수가 아니라 이 평가 점수로 합니다.
        if episode % eval_interval == 0:
            eval_score = evaluate_policy(policy_net, device)
            eval_episodes.append(episode)
            eval_scores.append(eval_score)
            print(f"  [평가] greedy 정책 5판 평균: {eval_score:.1f}")

            # 지금까지 중 가장 잘하는 시점의 가중치를 따로 저장해 둡니다.
            # (DQN은 학습이 출렁거려서 마지막 모델이 최고 모델이 아닐 수 있습니다)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                save_model(policy_net, best_model_path)

            if eval_score >= solved_threshold:
                print(f"greedy 평가 점수가 기준({solved_threshold:.0f}점)을 넘어 학습을 조기 종료합니다.")
                break

    env.close()
    save_model(policy_net, model_path)
    save_training_log(log_dir, scores, avg_scores, losses, epsilons, eval_episodes, eval_scores)
    plot_training_log(log_dir, scores, avg_scores, eval_episodes, eval_scores, solved_threshold)


if __name__ == "__main__":
    train()
