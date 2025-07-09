# DDPG + Hindsight Experience Replay (HER)

MuJoCo 기반 로봇팔 환경에서 DDPG와 HER 알고리즘을 적용해, 여러 개의 객체를 목표 지점에 옮기는 강화학습 프로젝트입니다. 희소한 보상 문제를 해결하고 샘플 효율성을 향상시키는 것을 목표로 합니다.

---

## 프로젝트 목표
- 두 개의 테이블로 구성된 환경 설정
- 오른쪽 테이블에 객체(Box) 10개 생성
- 왼쪽 테이블의 목표 지점(Target)으로 객체를 옮기기
- 환경 수정 파일(`fetch_env.py`, `pick_and_place.xml`)을 통해 복잡한 목표 설정 구현

---

## 사용 기술 및 알고리즘

### DDPG (Deep Deterministic Policy Gradient)
DDPG는 연속적인 행동 공간에서 동작하는 off-policy 기반의 강화학습 알고리즘입니다. Actor-Critic 구조를 기반으로 하며, 다음과 같은 기술을 적용했습니다:

- Ornstein-Uhlenbeck Process: 탐색을 위한 노이즈 적용
- Target Network Soft Update: 안정적인 학습을 위한 느린 업데이트
- Batch Normalization: 네트워크 안정화

### HER (Hindsight Experience Replay)
HER은 희소한 보상 환경에서 학습 효율을 높이기 위한 리플레이 전략입니다.

- 목표에 도달하지 못한 경험을 다른 목표로 치환하여 보상 신호를 생성
- 하나의 episode에서 여러 goal을 샘플링해 replay buffer에 추가

### 환경 커스터마이징
기존의 `FetchPickAndPlace-v2` 환경을 수정해, 다중 객체와 목표가 있는 시나리오를 구현했습니다.

- `pick_and_place.xml`: 두 개의 테이블, target 영역, 객체 박스 구성
- `fetch_env.py`: 목표 위치 샘플링 및 객체 배치 로직 수정, 종료 조건 재정의

---

## 실행 환경

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

## 학습 방법

```bash
mpirun -np 8 python -u train.py
```

---

## 테스트 방법
1. 테스트할 모델을 `saved_models/submitFirst/model.pt`에 저장
2. 아래 명령어로 실행

```bash
python demo.py --save-file submitFirst
```

---

## 학습 결과

학습 성공률 (2000 epoch 기준)
![학습데이터](./figures/RL_successrate.png)

시각화 결과 (GIF)

| 작은 경계 | 큰 경계 |
|-----------|-----------|
| ![small](./figures/smalloutput.gif) | ![big](./figures/8m52soutput.gif) |

---

## 폴더 구조

```
.
├── her_modules/         # HER 관련 모듈
├── rl_modules/          # DDPG 학습 모듈
├── saved_models/        # 저장된 모델
├── mpi_utils/           # MPI 유틸리티
├── figures/             # 학습 결과 이미지 및 애니메이션
├── train.py             # 학습 스크립트
├── demo.py              # 테스트 스크립트
├── fetch_env.py         # 테스트용 환경 파일
├── original_fetch_env.py# 학습용 환경 파일
├── pick_and_place.xml   # MuJoCo 환경 설정 파일
```

---

## 팀 정보
- 팀명: Ultron
- 팀원: 구현규, 하대훈

---

## 참고 자료
- [Hindsight Experience Replay, Andrychowicz et al. (2017)](https://arxiv.org/abs/1707.01495)
- [DDPG, Lillicrap et al. (2015)](https://arxiv.org/abs/1509.02971)


## 📚 참고 논문

- [Hindsight Experience Replay (Andrychowicz et al., 2017)](https://arxiv.org/abs/1707.01495)
- [DDPG (Lillicrap et al., 2015)](https://arxiv.org/abs/1509.02971)





