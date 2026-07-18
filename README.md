# DQN CartPole

DQN(Deep Q-Network)으로 [Gymnasium](https://gymnasium.farama.org/)의 `CartPole-v1` 환경을 학습하는 실습 예제입니다.
학부 인턴 세미나 실습용으로 만들었습니다.

## 구성

| 파일 | 설명 |
|---|---|
| `train.py` | DQN 학습 스크립트. 학습 후 가중치·로그·그래프를 저장합니다. |
| `evaluate.py` | 학습된 모델을 불러와 렌더링 창에서 성능을 확인합니다. |

핵심 구성 요소:

- **Replay Buffer** — 과거 경험을 저장하고 랜덤 샘플링해 샘플 간 상관관계를 줄입니다.
- **Epsilon-greedy** — 초반에는 탐험 위주, 학습이 진행될수록 활용 위주로 행동을 선택합니다.
- **Target Network** — 타깃 Q값 계산용 네트워크를 주기적으로만 갱신해 학습을 안정화합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 학습 (최근 20판 평균 475점 이상이면 조기 종료)
python train.py

# 학습된 모델 평가 (렌더링 창이 열립니다)
python evaluate.py
```

학습이 끝나면 다음 파일이 생성됩니다.

- `saved_models/dqn_cartpole_policy.pth` — 학습된 policy network 가중치
- `training_logs/training_log.csv` — 에피소드별 점수/손실/epsilon 로그
- `training_logs/training_plot.png` — 점수·손실 그래프
