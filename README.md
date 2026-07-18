# DQN CartPole

DQN(Deep Q-Network)으로 [Gymnasium](https://gymnasium.farama.org/)의 `CartPole-v1` 환경을 학습하는 실습 예제입니다.
학부 인턴 세미나 실습용으로 만들었습니다.

## 구성

| 파일 | 설명 |
|---|---|
| `train.py` | DQN 학습 스크립트. 학습 후 가중치·로그·그래프를 저장합니다. |
| `evaluate.py` | 학습된 모델(best 우선)을 불러와 렌더링 창에서 성능을 확인합니다. |

핵심 구성 요소:

- **Replay Buffer** — 과거 경험을 저장하고 랜덤 샘플링해 샘플 간 상관관계를 줄입니다.
- **Epsilon-greedy** — 초반에는 탐험 위주, 학습이 진행될수록 활용 위주로 행동을 선택합니다.
- **Target Network** — 타깃 Q값 계산용 네트워크를 주기적으로만 갱신해 학습을 안정화합니다.
- **Greedy 평가** — 10 에피소드마다 탐험 없이(greedy) 5판을 돌려 평균 점수를 잽니다.
  조기 종료(최대 점수의 95% = 475점)와 best 모델 저장은 이 평가 점수로 판단합니다.

## 평가지표에 대해

- **학습 점수(왼쪽 그래프)** 는 epsilon-greedy 행동 정책의 점수라 무작위 탐험이 섞여 있어
  정책의 실제 실력보다 낮게 나옵니다. 학습 흐름을 보는 참고용입니다.
- **평가 점수(오른쪽 그래프)** 는 greedy 정책만으로 돌린 점수로, 정책의 실제 실력은
  이 곡선으로 판단합니다.
- **TD loss는 성능 지표가 아닙니다.** 타깃이 계속 움직이는 자기 일관성 오차라서,
  학습이 잘 될수록 오히려 커질 수 있습니다. 발산 진단용으로 CSV에만 기록합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 학습 (greedy 평가 평균이 475점 이상이면 조기 종료)
python train.py

# 학습된 모델 평가 (렌더링 창이 열립니다)
python evaluate.py
```

학습이 끝나면 다음 파일이 생성됩니다.

- `saved_models/dqn_cartpole_best.pth` — 평가 점수가 가장 높았던 시점의 가중치
- `saved_models/dqn_cartpole_policy.pth` — 마지막 에피소드의 가중치
- `training_logs/training_log.csv` — 에피소드별 점수/손실/epsilon 로그
- `training_logs/eval_log.csv` — greedy 평가 점수 로그
- `training_logs/training_plot.png` — 학습 점수·평가 점수 그래프
