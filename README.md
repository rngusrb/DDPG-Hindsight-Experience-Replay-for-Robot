# 🦾 DDPG + Hindsight Experience Replay (HER)

MuJoCo 기반 로봇팔 환경에서 DDPG + HER를 적용하여 **10개의 객체를 목표 지점에 옮기는 강화학습 프로젝트**입니다. 학습 환경과 알고리즘 튜닝을 통해 sparse reward 문제를 해결하고, 높은 샘플 효율성을 달성하는 것이 목표입니다.

---

## 📌 주요 목표
- 두 개의 테이블 환경 구성
- 한쪽 테이블에서 객체(Box) 10개 생성
- 다른 테이블에 생성된 목표 지점(Target)에 객체들을 옮기기
- 복잡한 목표 세팅을 위한 `fetch_env.py`, `pick_and_place.xml` 커스터마이징

---

## 🚀 사용 기술 및 알고리즘

### 🧠 DDPG (Deep Deterministic Policy Gradient)

DDPG는 **연속적인 행동 공간**을 다룰 수 있는 Off-policy 강화학습 알고리즘입니다. DQN의 한계를 극복하고, Actor-Critic 구조로 설계되어 있습니다.

- **Actor Network**: 현재 상태에서 어떤 행동을 취할지 결정 (`π(s)`)
- **Critic Network**: 해당 행동의 Q-value를 평가 (`Q(s, a)`)

#### 🔧 사용한 핵심 기법들
- **Ornstein-Uhlenbeck Process (OU Noise)**: 연속 공간에서 exploration 강화를 위한 noise 방식
- **Target Network & Soft Update**:
  ```
  θ_target ← τθ_online + (1 - τ)θ_target (τ << 1)
  ```
- **Batch Normalization**: 네트워크 학습 안정성을 높이기 위한 정규화 기법

---

### 🎯 Hindsight Experience Replay (HER)

HER은 **희소한 보상(sparse reward)** 문제를 해결하기 위한 경험 리플레이 전략입니다.

#### 📌 핵심 개념
- 기존 goal로 reward를 못 받은 transition을 → 다른 goal로 바꿔서 reward를 받을 수 있게 만드는 방식
- 동일한 episode 내에서 추가 goal을 샘플링하여 reward를 재계산한 후 replay buffer에 추가

#### 💡 적용 과정
1. DDPG를 기반으로 학습 진행
2. Replay Buffer에 실제 goal 외에 추가적인 goal들(g')을 포함
3. reward 함수는 g' 기준으로 다시 계산하여 업데이트

---

### 🛠 환경 커스터마이징

기존 `gymnasium-robotics`의 `FetchPickAndPlace-v2`를 기반으로, 복잡한 다중 목표 설정을 위해 환경을 커스터마이징했습니다.

#### 수정 사항
- `pick_and_place.xml`:
  - 테이블 2개 생성
  - 좌측 테이블: target zone (site)
  - 우측 테이블: object 박스 10개 생성
- `fetch_env.py`, `original_fetch_env.py`:
  - `reset_sim()` 수정 → 10개 박스 랜덤 배치
  - `sample_goal()` 수정 → target 위치 다양화
  - `step()` 오버라이딩 → 에피소드 종료 조건 추가 (10개 옮기면 종료)

---

## ⚙️ 실행 환경

```bash
python==3.11
mujoco==2.3.7
mpi4py==3.1.4
gym==0.26.2
gymnasium==0.29.1
gymnasium-robotics==1.2.4
numpy==1.26.4
torch==2.3.0
```

---

## 🏋️‍♂️ 학습

```bash
# 멀티 프로세스 학습 (8개 프로세스로)
mpirun -np 8 python -u train.py
```

---

## 🎯 테스트

1. 테스트할 모델을 `/saved_models/` 하위에 저장 (예: `submitFirst/model.pt`)
2. 아래와 같이 실행

```bash
python demo.py --save-file submitFirst
```

---

## 📈 학습 결과

### Success Rate (2000 Epoch)
![학습데이터](./figures/RL_successrate.png)

### 시각화 결과 (GIF)

| 작은 경계 | 큰 경계 |
|:--:|:--:|
| ![small](./figures/smalloutput.gif) | ![big](./figures/8m52soutput.gif) |

---

## 📁 폴더 구조 요약

```
.
├── her_modules/         # HER 관련 모듈
├── rl_modules/          # DDPG 학습 모듈
├── saved_models/        # 저장된 모델
├── mpi_utils/           # MPI 병렬처리 유틸
├── figures/             # 학습 결과 이미지, GIF
├── train.py             # 학습 스크립트
├── demo.py              # 테스트 스크립트
├── fetch_env.py         # 테스트용 환경
├── original_fetch_env.py# 학습용 환경
├── pick_and_place.xml   # Mujoco 환경 설정
```

---

## 🙋 팀 정보

- 팀명: Ultron  
- 팀원: 구현규, 하대훈

---

## 📚 참고 논문

- [Hindsight Experience Replay (Andrychowicz et al., 2017)](https://arxiv.org/abs/1707.01495)
- [DDPG (Lillicrap et al., 2015)](https://arxiv.org/abs/1509.02971)

## 학습결과
 2000 epoch동안 학습시 success rate
![학습데이터](./figures/RL_successrate.png)
### small border
![small](./figures/smalloutput.gif)



### big border
![big](./figures/8m52soutput.gif)



