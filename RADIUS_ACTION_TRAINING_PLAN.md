# Radius-Action Neighbor Selection Training Plan

## Goal

기존 neighbor-selection 학습은 각 agent가 가능한 모든 pairwise neighbor edge를 `N x N` binary action으로 선택하는 구조였다. 이번 실험의 목표는 action space를 단순화하여, 각 agent가 scalar interaction range 하나만 출력하게 만드는 것이다.

새 정책은 기존과 같은 observation을 입력으로 받되, 출력은 agent별 normalized disk radius가 된다. 환경은 각 agent `i`에 대해 `dist(i, j) <= r_i`인 neighbor만 flocking controller에 전달한다.

## Current Baseline

현재 구조:

- Input: `local_agent_infos`, `neighbor_masks`, `padding_mask`
- Action: `(num_agents_max, num_agents_max)` binary matrix
- Meaning: `action[i, j] = 1`이면 agent `i`가 agent `j`의 정보를 사용
- Control path: `state["neighbor_masks"] & action`으로 selected network를 만들고 ACS/Vicsek control에 전달

새 구조:

- Input: 기존 observation과 동일
- Action: `(num_agents_max,)` continuous vector
- Meaning: `action[i]`는 agent `i`의 normalized interaction range
- Control path: physical radius로 변환한 뒤 row-wise disk mask를 만들고 ACS/Vicsek control에 전달

## Action Contract

권장 action space:

```text
action_space = Box(low=0.0, high=1.0, shape=(num_agents_max,), dtype=float32)
```

각 action 값을 physical radius로 변환한다.

```text
a_i in [0, 1]
r_i = r_min + a_i * (r_max - r_min)
```

초기 실험에서는 다음 설정을 추천한다.

```text
r_min = 0
r_max = initial_position_bound
```

이후 ablation 후보:

```text
r_min = 0.2 * r0, r_max = 4 * r0
r_min = r0,       r_max = 4 * r0
```

현재 기본값 기준 `r0 = 60`, `initial_position_bound = 250`이므로 `4 * r0 = 240`은 전체 환경 scale과 거의 맞다.

## Disk Mask Semantics

각 agent `i`에 대해 선택된 neighbor mask는 다음과 같이 만든다.

```text
disk_mask[i, j] = dist(i, j) <= r_i
selected_network = neighbor_masks & disk_mask & padding_mask_2d
```

정책:

- Self-loop는 radius와 무관하게 항상 포함한다.
- Padding agent의 action은 무시한다.
- `comm_range is None`이면 후보 neighbor는 전체 active flock이다.
- `comm_range`가 설정되어 있으면 `neighbor_masks & disk_mask`로 물리 통신 가능 범위를 먼저 제한한다.

## Environment Plan

코드 구현 시 예상 변경 범위:

- `EnvConfig.action_type`에 `"radius"` 사용
- `NeighborSelectionFlockingEnv.__init__`에서 radius action space 허용
- `to_binary_action()`에서 normalized radius action을 disk mask로 변환
- radius action logging을 위해 `info`에 mean/max/min radius와 selected neighbor count 추가
- 기존 `binary_vector` action path는 유지

검증할 조건:

- `action.shape == (num_agents_max,)`
- action 값은 env 내부에서 `[0, 1]`로 clip 또는 assert
- active agent diagonal은 항상 1
- padding row/column은 선택되지 않음
- `comm_range`가 있을 때는 그 밖의 neighbor를 선택하지 않음

## Neural Network Plan

입력은 기존 ego-centric observation을 그대로 사용한다.

```text
local_agent_infos: (batch, N, N, obs_dim)
neighbor_masks:    (batch, N, N)
padding_mask:      (batch, N)
```

출력은 agent별 scalar action 분포다.

```text
policy output: mean/log_std for N scalar actions
sampled action: (batch, N)
```

권장 구조:

1. 각 focal agent `i`의 local neighbor tokens `local_agent_infos[:, i, :, :]`를 encoder에 입력
2. `neighbor_masks[:, i, :]`와 `padding_mask`로 invalid token masking
3. masked mean pooling 또는 attention pooling으로 agent context `h_i` 생성
4. scalar MLP head로 radius action의 distribution parameter 출력

예상 구조:

```text
local tokens -> embedding -> transformer encoder -> masked pooling -> MLP -> radius mean
```

PPO continuous action에서 선택할 수 있는 방식:

- 단순안: Gaussian mean/log_std 출력 후 env에서 action을 `[0, 1]`로 clip
- 권장안: squashed Gaussian 또는 sigmoid transform으로 action을 `[0, 1]`에 매핑

초기 구현은 단순안을 사용해 smoke test를 빠르게 통과시키고, action이 boundary에 자주 붙으면 squashed distribution으로 개선한다.

## Reward Plan

1차 실험에서는 기존 ACS training reward를 유지한다.

이유:

- action space 변경 자체의 효과를 먼저 봐야 한다.
- range penalty를 바로 넣으면 수렴 성능과 sparse interaction 선호가 섞여 해석이 어려워진다.

2차 실험에서 range regularization을 추가한다.

```text
reward = existing_training_reward - w_range * mean(a_i)
```

후보:

```text
w_range in {0.001, 0.005, 0.01, 0.02}
```

range penalty를 넣을 때는 반드시 success rate와 convergence step을 함께 봐야 한다. reward만 보면 작은 radius를 쓰는 정책이 실제 flocking 수렴을 희생할 수 있다.

## Baselines

새 action type을 평가하려면 다음 baseline을 먼저 준비한다.

- All-neighbor: 모든 active neighbor 사용
- Fixed radius `r = r0`
- Fixed radius `r = 2 * r0`
- Fixed radius `r = 4 * r0`
- Fixed normalized radius `a = 0.25, 0.5, 0.75, 1.0`
- Existing binary neighbor-selection policy

이 baseline들은 learned radius policy가 단순히 큰 radius를 항상 선택하는지, 또는 실제로 상황별 range 조절을 학습하는지 확인하는 기준이 된다.

## Evaluation Metrics

이번 실험의 핵심은 reward보다 convergence time이다. 별도 evaluator에서 아래 값을 episode별로 저장한다.

- `first_converged_step`
- `success`
- `success_rate`
- `episode_length`
- `episode_return`
- `original_episode_return`
- final spatial entropy
- final velocity entropy
- mean selected neighbor count
- mean radius
- std radius
- radius histogram
- selected neighbor count histogram

ACS convergence 기준 후보:

```text
spatial_entropy < entropy_p_goal
velocity_entropy < entropy_v_goal
stable over entropy_rate_window_length
```

고정 길이 episode에서도 `first_converged_step`은 별도로 기록한다. 수렴하지 못한 episode는 `first_converged_step = max_time_steps` 또는 `NaN`으로 처리하되, 집계 방식은 문서에 명시한다.

## Experiment Phases

### Phase 0: Contract Finalization

- `r_min`, `r_max` 정의 확정
- `comm_range=None`과 finite `comm_range`에서 disk mask 의미 확정
- action clipping과 self-loop 정책 확정

권장 결정:

```text
r_min = 0
r_max = initial_position_bound
self-loop always selected
selected_network = neighbor_masks & disk_mask & padding_mask_2d
```

### Phase 1: Environment Smoke Test

- radius action을 binary disk mask로 변환하는 path 검증
- random radius action rollout
- fixed radius action rollout
- padding/mask/self-loop validation
- mean radius와 selected neighbor count가 기대대로 기록되는지 확인

### Phase 2: Model Smoke Test

- radius policy model forward pass 검증
- output shape 확인
- RLlib action distribution과 action space compatibility 확인
- one batch PPO update가 에러 없이 도는지 확인

### Phase 3: Small Training

작은 설정에서 빠르게 학습 가능성을 본다.

```text
num_agents_pool = [5] or [10]
max_time_steps = 300
num_workers = 1
train_batch_size small
```

확인할 것:

- reward가 NaN 없이 움직이는가
- action이 항상 0 또는 1에 붙지 않는가
- radius 평균이 학습 중 변화하는가
- selected neighbor count가 reasonable한가

### Phase 4: Full Training

기존 실험과 비교 가능한 설정으로 확장한다.

```text
num_agents_pool = [20]
max_time_steps = 1000
task_type = acs
observation_type = ego_centric
```

checkpoint는 latest가 아니라 validation metric 기준 best를 따로 고른다.

### Phase 5: Evaluation

동일 seed set에서 아래 policy들을 비교한다.

- All-neighbor
- Fixed radius baselines
- Learned radius policy
- Existing binary neighbor-selection policy
- Selected heuristic baselines

집계는 평균만 보지 말고 success rate와 confidence interval을 함께 본다.

## Risks And Checks

- Continuous action을 단순 clip하면 policy가 boundary에 붙을 수 있다.
- `r_max`가 너무 크면 all-neighbor와 같아질 수 있다.
- `r_min=0`이면 초반 exploration에서 isolated behavior가 많아질 수 있다.
- reward shaping이 convergence time과 완전히 일치하지 않을 수 있다.
- 현재 저장 결과는 training reward와 original reward가 크게 다르므로 비교 시 단위를 분리해야 한다.
- 현재 `evaluate_checkpoint.py`는 centralized path와 ego-centric checkpoint가 섞일 수 있으므로 radius experiment 전에 평가 스크립트를 정리해야 한다.

## Recommended First Configuration

```text
action_type = radius
observation_type = ego_centric
num_agents_pool = [20]
max_time_steps = 1000
r_min = 0
r_max = initial_position_bound
range_penalty = 0
use_fixed_episode_length = True
```

첫 학습의 목적은 성능 최적화가 아니라 action-space feasibility 확인이다. 성공 기준은 다음과 같다.

- PPO가 안정적으로 rollout/update 수행
- action radius가 non-trivial distribution을 가짐
- selected neighbor count가 all-neighbor보다 작아지는 episode가 있음
- final entropy와 convergence metric이 fixed-radius baseline과 비교 가능

## Open Decisions

- `r_max`를 `initial_position_bound`로 둘지 `4 * r0`로 둘지
- `r_min=0`을 허용할지
- continuous action distribution을 clip 기반으로 시작할지 squashed Gaussian으로 바로 갈지
- range penalty를 언제부터 넣을지
- evaluation에서 미수렴 episode의 convergence step을 `max_time_steps`로 둘지 `NaN`으로 둘지
