# GR00T Inference Code Flow Analysis (TensorRT Mode)

## 개요
이 문서는 `python deployment_scripts/gr00t_inference.py --inference-mode=tensorrt` 실행 시의 코드 흐름을 분석합니다.

## 전체 실행 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 초기화 단계                                                    │
│    - 인자 파싱 (--inference-mode=tensorrt)                        │
│    - 경로 설정 (model_path, dataset_path, trt_engine_path)       │
│    - 데이터 설정 로드 (data_config, modality_config)             │
│    - Gr00tPolicy 초기화                                           │
│      ├─ 모델 다운로드/로드 (GR00T_N1_5)                          │
│      ├─ 메타데이터 로드                                           │
│      └─ Horizon 설정                                              │
│    - 데이터셋 로드 (LeRobotSingleDataset)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. TensorRT 엔진 설정                                            │
│    setup_tensorrt_engines()                                      │
│    ├─ PyTorch 모델 삭제 (메모리 절약)                            │
│    ├─ TensorRT 엔진 로드                                         │
│    │  ├─ vit_fp8.engine (Vision Transformer)                    │
│    │  ├─ llm_nvfp4.engine (Language Model)                      │
│    │  ├─ vlln_vl_self_attention.engine                          │
│    │  ├─ state_encoder.engine                                    │
│    │  ├─ action_encoder.engine                                  │
│    │  ├─ DiT_fp8.engine (Diffusion Transformer)                 │
│    │  └─ action_decoder.engine                                   │
│    └─ Forward 함수 교체                                           │
│       ├─ backbone.forward → eagle_tensorrt_forward                │
│       └─ action_head.get_action → action_head_tensorrt_forward  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 추론 실행                                                      │
│    policy.get_action(step_data)                                  │
│    ├─ 입력 전처리 (배치 차원 추가, numpy 변환)                    │
│    ├─ Transform 적용 (정규화)                                    │
│    └─ 모델 추론                                                   │
│       └─ model.get_action()                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Backbone 추론 (eagle_tensorrt_forward)                        │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ ViT Engine 실행                                       │     │
│    │   pixel_values → vit_embeds                           │     │
│    └──────────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ Pixel Shuffle + MLP 처리                              │     │
│    └──────────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ Vision-Language 결합                                 │     │
│    │   input_ids + vit_embeds → input_embeds              │     │
│    └──────────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ LLM Engine 실행                                       │     │
│    │   input_embeds → backbone_features                    │     │
│    └──────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Action Head 추론 (action_head_tensorrt_forward)              │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ VL Self-Attention Engine                             │     │
│    │   backbone_features → vl_embs                        │     │
│    └──────────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ State Encoder Engine                                  │     │
│    │   state + embodiment_id → state_features              │     │
│    └──────────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ 초기 액션 설정 (랜덤 노이즈)                          │     │
│    └──────────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │ Denoising Loop (4 steps)                              │     │
│    │   for t in range(num_steps):                          │     │
│    │     ├─ Action Encoder Engine                          │     │
│    │     │    actions → action_features                    │     │
│    │     ├─ Position Embedding 추가                        │     │
│    │     ├─ State + Future + Action 결합                  │     │
│    │     │    → sa_embs                                    │     │
│    │     ├─ DiT Engine 실행                                │     │
│    │     │    (sa_embs, vl_embs) → model_output            │     │
│    │     ├─ Action Decoder Engine                          │     │
│    │     │    model_output → pred_velocity                 │     │
│    │     └─ Euler Integration                              │     │
│    │        actions = actions + dt * pred_velocity         │     │
│    └──────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. 후처리                                                         │
│    ├─ 정규화 해제 (unapply_transforms)                           │
│    ├─ 배치 차원 제거 (단일 샘플인 경우)                          │
│    └─ 결과 반환                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. 결과 출력                                                      │
│    === TensorRT Inference Results ===                            │
│    action (4, 14)  # action_horizon=4, action_dim=14            │
└─────────────────────────────────────────────────────────────────┘
```

## 전체 실행 흐름

### 1. 초기화 단계 (gr00t_inference.py:93-243)

#### 1.1 인자 파싱 (라인 163)
```python
args = parser.parse_args()
```
- `--inference-mode=tensorrt` 설정
- 기본값들:
  - `model_path`: "nvidia/GR00T-N1.5-3B"
  - `trt_engine_path`: "gr00t_engine"
  - `vit_dtype`: "fp8"
  - `llm_dtype`: "nvfp4"
  - `dit_dtype`: "fp8"
  - `denoising_steps`: 4

#### 1.2 경로 설정 (라인 165-172)
```python
MODEL_PATH = args.model_path
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = args.dataset_path or os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
EMBODIMENT_TAG = args.embodiment_tag  # 기본값: "gr1"
```

#### 1.3 데이터 설정 로드 (라인 177-181)
```python
from gr00t.experiment.data_config import load_data_config
data_config = load_data_config(args.data_config)  # 기본값: "fourier_gr1_arms_only"
modality_config = data_config.modality_config()
modality_transform = data_config.transform()
```

#### 1.4 Policy 초기화 (라인 183-190)
```python
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    denoising_steps=args.denoising_steps,
    device=device,
)
```

**Gr00tPolicy.__init__ 내부 흐름:**
1. **모델 다운로드/로드** (policy.py:85-93)
   - HuggingFace Hub에서 모델 다운로드 시도
   - 실패 시 로컬 경로 사용

2. **모델 로드** (policy.py:239-277)
   ```python
   model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
   model.eval()
   model.to(device=self.device)
   ```
   - `GR00T_N1_5` 모델 인스턴스 생성
   - Backbone (Eagle) + Action Head (Flowmatching) 구조

3. **메타데이터 로드** (policy.py:279-297)
   - `experiment_cfg/metadata.json`에서 정규화 통계 로드
   - Embodiment별 메타데이터 설정

4. **Horizon 설정** (policy.py:299-313)
   - Video/State delta indices 설정

#### 1.5 데이터셋 로드 (라인 193-200)
```python
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend=args.video_backend,
    ...
)
step_data = dataset[0]  # 첫 번째 샘플 로드
```

### 2. TensorRT 엔진 설정 단계 (라인 210-214)

```python
setup_tensorrt_engines(
    policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
)
```

**setup_tensorrt_engines 내부 흐름 (trt_model_forward.py:195-267):**

#### 2.1 모델 구조 정리 (라인 208-235)
```python
# Backbone에서 필요한 속성 저장
policy.model.backbone.num_patches = ...
policy.model.backbone.embedding_layer = ...
policy.model.backbone.image_token_index = ...

# PyTorch 모델 삭제 (메모리 절약)
del policy.model.backbone.eagle_model.vision_model
del policy.model.backbone.eagle_model.language_model
del policy.model.action_head.vlln
del policy.model.action_head.vl_self_attention
del policy.model.action_head.model
del policy.model.action_head.state_encoder
del policy.model.action_head.action_encoder
del policy.model.action_head.action_decoder
torch.cuda.empty_cache()
```

#### 2.2 TensorRT 엔진 로드 (라인 238-260)
```python
# Backbone 엔진들
policy.model.backbone.vit_engine = trt.Engine(
    os.path.join(trt_engine_path, f"vit_{vit_dtype}.engine")  # vit_fp8.engine
)
policy.model.backbone.llm_engine = trt.Engine(
    os.path.join(trt_engine_path, f"llm_{llm_dtype}.engine")  # llm_nvfp4.engine
)

# Action Head 엔진들
policy.model.action_head.vlln_vl_self_attention_engine = trt.Engine(
    os.path.join(trt_engine_path, "vlln_vl_self_attention.engine")
)
policy.model.action_head.action_encoder_engine = trt.Engine(
    os.path.join(trt_engine_path, "action_encoder.engine")
)
policy.model.action_head.action_decoder_engine = trt.Engine(
    os.path.join(trt_engine_path, "action_decoder.engine")
)
policy.model.action_head.DiT_engine = trt.Engine(
    os.path.join(trt_engine_path, f"DiT_{dit_dtype}.engine")  # DiT_fp8.engine
)
policy.model.action_head.state_encoder_engine = trt.Engine(
    os.path.join(trt_engine_path, "state_encoder.engine")
)
```

**trt.Engine.__init__ 내부 (trt_torch.py:42-58):**
1. TensorRT Runtime 초기화
2. 엔진 파일 역직렬화
3. Execution Context 생성
4. Input/Output 메타데이터 추출
5. 엔진 정보 출력 (라인 58-75)

#### 2.3 Forward 함수 교체 (라인 262-266)
```python
# Backbone forward를 TensorRT 버전으로 교체
policy.model.backbone.forward = partial(eagle_tensorrt_forward, policy.model.backbone)

# Action Head get_action을 TensorRT 버전으로 교체
policy.model.action_head.get_action = partial(
    action_head_tensorrt_forward, policy.model.action_head
)
```

### 3. 추론 실행 단계 (라인 216-219)

```python
predicted_action = policy.get_action(step_data)
```

**policy.get_action 내부 흐름 (policy.py:146-186):**

#### 3.1 입력 전처리 (라인 168-179)
```python
obs_copy = observations.copy()
is_batch = self._check_state_is_batched(obs_copy)
if not is_batch:
    obs_copy = unsqueeze_dict_values(obs_copy)  # 배치 차원 추가

# numpy 배열로 변환
for k, v in obs_copy.items():
    if not isinstance(v, np.ndarray):
        obs_copy[k] = np.array(v)
```

#### 3.2 Transform 적용 (라인 180)
```python
normalized_input = self.apply_transforms(obs_copy)
```
- 비디오 정규화
- 상태 정규화
- 어노테이션 처리

#### 3.3 모델 추론 (라인 181)
```python
normalized_action = self._get_action_from_normalized_input(normalized_input)
```

**내부 흐름 (policy.py:188-194):**
```python
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
    model_pred = self.model.get_action(normalized_input)
```

**model.get_action 내부 (gr00t_n1.py:171-180):**
```python
backbone_inputs, action_inputs = self.prepare_input(inputs)
backbone_outputs = self.backbone(backbone_inputs)  # ← TensorRT 실행!
action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)  # ← TensorRT 실행!
```

### 4. Backbone TensorRT 추론 (eagle_tensorrt_forward)

**eagle_tensorrt_forward 흐름 (trt_model_forward.py:26-101):**

#### 4.1 입력 준비 (라인 26-38)
```python
# Eagle prefix 제거
eagle_input = {k.removeprefix("eagle_"): v for k, v in vl_input.items() if k.startswith("eagle_")}
del eagle_input["image_sizes"]
vl_input = eagle_input

# 배치 크기 확인 (최대 8)
batch_size = vl_input["pixel_values"].shape[0]
assert batch_size <= 8

# float16 변환
if vl_input["pixel_values"].dtype != torch.float16:
    vl_input["pixel_values"] = vl_input["pixel_values"].to(torch.float16)
```

#### 4.2 ViT 엔진 실행 (라인 44-46)
```python
position_ids = torch.arange(self.num_patches, device="cuda").expand((batch_size, -1))

self.vit_engine.set_runtime_tensor_shape("pixel_values", vl_input["pixel_values"].shape)
self.vit_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)
vit_embeds = self.vit_engine(vl_input["pixel_values"], position_ids)["vit_embeds"]
```

**trt.Engine.forward 내부 (trt_torch.py:102-162):**
1. 입력 텐서 검증 (shape, dtype, device)
2. CUDA 메모리 주소 설정
3. 출력 텐서 메모리 할당
4. `execute_async_v3` 실행
5. CUDA stream 동기화
6. 결과 반환

#### 4.3 Pixel Shuffle 처리 (라인 49-57)
```python
if self.eagle_model.use_pixel_shuffle:
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
```

#### 4.4 MLP 처리 (라인 59-62)
```python
if self.eagle_model.mlp_checkpoint and vit_embeds.requires_grad:
    vit_embeds = cp.checkpoint(self.eagle_model.mlp1, vit_embeds)
else:
    vit_embeds = self.eagle_model.mlp1(vit_embeds)
```

#### 4.5 텍스트 임베딩 준비 (라인 64-90)
```python
input_ids = vl_input["input_ids"]
input_embeds = self.embedding_layer(input_ids)  # PyTorch embedding

# float16 변환
if input_embeds.dtype != torch.float16:
    input_embeds = input_embeds.to(torch.float16)

# Vision-Language 결합
B, N, C = input_embeds.shape
input_embeds = input_embeds.reshape(B * N, C)
input_ids_flat = input_ids.reshape(B * N)
selected = input_ids_flat == self.image_token_index
input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
input_embeds = input_embeds.reshape(B, N, C)
```

#### 4.6 LLM 엔진 실행 (라인 92-94)
```python
self.llm_engine.set_runtime_tensor_shape("inputs_embeds", input_embeds.shape)
self.llm_engine.set_runtime_tensor_shape("attention_mask", vl_input["attention_mask"].shape)
embeddings = self.llm_engine(input_embeds, vl_input["attention_mask"])["embeddings"]
```

#### 4.7 결과 반환 (라인 96-101)
```python
return BatchFeature(
    data={
        "backbone_features": embeddings,
        "backbone_attention_mask": vl_input["attention_mask"],
    }
)
```

### 5. Action Head TensorRT 추론 (action_head_tensorrt_forward)

**action_head_tensorrt_forward 흐름 (trt_model_forward.py:104-192):**

#### 5.1 VL Self-Attention (라인 106-114)
```python
if backbone_output.backbone_features.dtype != torch.float16:
    backbone_output.backbone_features = backbone_output.backbone_features.to(torch.float16)

self.vlln_vl_self_attention_engine.set_runtime_tensor_shape(
    "backbone_features", backbone_output.backbone_features.shape
)
backbone_output.backbone_features = self.vlln_vl_self_attention_engine(
    backbone_output.backbone_features
)["output"]
vl_embs = backbone_output.backbone_features
```

#### 5.2 State 인코딩 (라인 115-131)
```python
embodiment_id = action_input.embodiment_id
batch_size = vl_embs.shape[0]

# dtype 변환
if action_input.state.dtype != torch.float16:
    action_input.state = action_input.state.to(torch.float16)
if embodiment_id.dtype != torch.int64:
    embodiment_id = embodiment_id.to(int64)

# State Encoder 엔진 실행
self.state_encoder_engine.set_runtime_tensor_shape("state", action_input.state.shape)
self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
state_features = self.state_encoder_engine(action_input.state, embodiment_id)["output"]
```

#### 5.3 초기 액션 설정 (라인 133-144)
```python
device = vl_embs.device
if hasattr(self, "init_actions"):
    actions = self.init_actions.expand((batch_size, -1, -1))
else:
    actions = torch.randn(
        size=(batch_size, self.config.action_horizon, self.config.action_dim),
        dtype=vl_embs.dtype,
        device=device,
    )
```

#### 5.4 Denoising Loop (라인 146-191)
```python
num_steps = self.num_inference_timesteps  # 기본값: 4
dt = 1.0 / num_steps

for t in range(num_steps):
    t_cont = t / float(num_steps)  # 0, 0.25, 0.5, 0.75
    t_discretized = int(t_cont * self.num_timestep_buckets)
    
    # 5.4.1 Action 인코딩
    timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
    
    self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
    self.action_encoder_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
    self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    action_features = self.action_encoder_engine(actions, timesteps_tensor, embodiment_id)["output"]
    
    # 5.4.2 Position Embedding 추가
    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
        action_features = action_features + pos_embs
    
    # 5.4.3 State, Future, Action 임베딩 결합
    future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(torch.float16)
    
    # 5.4.4 DiT (Diffusion Transformer) 엔진 실행
    if vl_embs.dtype != torch.float16:
        vl_embs = vl_embs.to(torch.float16)
    
    self.DiT_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
    self.DiT_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
    self.DiT_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
    model_output = self.DiT_engine(sa_embs, vl_embs, timesteps_tensor)["output"]
    
    # 5.4.5 Action 디코딩
    self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
    self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
    pred_velocity = pred[:, -self.action_horizon :]
    
    # 5.4.6 Euler Integration으로 액션 업데이트
    actions = actions + dt * pred_velocity

# 5.5 결과 반환
return BatchFeature(data={"action_pred": actions})
```

### 6. 후처리 단계 (policy.py:182-186)

```python
# 정규화 해제
unnormalized_action = self._get_unnormalized_action(normalized_action)

# 배치 차원 제거 (단일 샘플인 경우)
if not is_batch:
    unnormalized_action = squeeze_dict_values(unnormalized_action)

return unnormalized_action
```

### 7. 결과 출력 (라인 217-219)

```python
print("\n=== TensorRT Inference Results ===")
for key, value in predicted_action.items():
    print(key, value.shape)
```

## 주요 컴포넌트 요약

### TensorRT 엔진들
1. **ViT Engine**: Vision Transformer (이미지 → 임베딩)
2. **LLM Engine**: Language Model (텍스트 + 이미지 → 통합 임베딩)
3. **VL Self-Attention Engine**: Vision-Language Self-Attention
4. **State Encoder Engine**: 상태 정보 인코딩
5. **Action Encoder Engine**: 액션 시퀀스 인코딩
6. **DiT Engine**: Diffusion Transformer (메인 생성 모델)
7. **Action Decoder Engine**: 액션 예측 디코딩

### 데이터 흐름
```
입력 (video, state, annotation)
  ↓
Transform (정규화)
  ↓
Backbone (ViT + LLM) → backbone_features
  ↓
VL Self-Attention → vl_embs
  ↓
State Encoder → state_features
  ↓
[Denoising Loop: 4 steps]
  ├─ Action Encoder → action_features
  ├─ DiT (sa_embs, vl_embs) → model_output
  ├─ Action Decoder → pred_velocity
  └─ Euler Integration → actions 업데이트
  ↓
최종 actions
  ↓
Unapply Transform (정규화 해제)
  ↓
출력 (action)
```

### 메모리 최적화
- PyTorch 모델 삭제 후 TensorRT 엔진만 사용
- `torch.cuda.empty_cache()` 호출
- float16/fp8 정밀도 사용

### 성능 최적화
- TensorRT 엔진의 비동기 실행 (`execute_async_v3`)
- CUDA stream 활용
- 배치 처리 지원 (최대 batch_size=8)

## 예상 터미널 로그

실행 시 다음과 같은 로그가 출력됩니다:

```
# 1. 모델 다운로드/로드 (HuggingFace Hub 또는 로컬)
Model not found or avail in the huggingface hub. Loading from local path: nvidia/GR00T-N1.5-3B
# 또는
Downloading model files: 100%|████████████| 12345/12345 [00:30<00:00, 411.50it/s]

# 2. Policy 초기화
Set action denoising steps to 4

# 3. TensorRT 엔진 로드 (각 엔진마다 출력)
============= TRT Engine Detail =============
Engine file: gr00t_engine/vit_fp8.engine
Inputs: 2
   0. pixel_values: 1x3x224x224 [torch.float16]
   1. position_ids: 1x196 [torch.int64]
Outputs: 1
   0. vit_embeds: 1x196x1152 [torch.float16]
=============================================

============= TRT Engine Detail =============
Engine file: gr00t_engine/llm_nvfp4.engine
Inputs: 2
   0. inputs_embeds: 1x256x4096 [torch.float16]
   1. attention_mask: 1x256 [torch.int64]
Outputs: 1
   0. embeddings: 1x256x4096 [torch.float16]
=============================================

============= TRT Engine Detail =============
Engine file: gr00t_engine/vlln_vl_self_attention.engine
Inputs: 1
   0. backbone_features: 1x256x4096 [torch.float16]
Outputs: 1
   0. output: 1x256x4096 [torch.float16]
=============================================

============= TRT Engine Detail =============
Engine file: gr00t_engine/state_encoder.engine
Inputs: 2
   0. state: 1x14 [torch.float16]
   1. embodiment_id: 1 [torch.int64]
Outputs: 1
   0. output: 1x1x4096 [torch.float16]
=============================================

============= TRT Engine Detail =============
Engine file: gr00t_engine/action_encoder.engine
Inputs: 3
   0. actions: 1x4x14 [torch.float16]
   1. timesteps_tensor: 1 [torch.int64]
   2. embodiment_id: 1 [torch.int64]
Outputs: 1
   0. output: 1x4x4096 [torch.float16]
=============================================

============= TRT Engine Detail =============
Engine file: gr00t_engine/DiT_fp8.engine
Inputs: 3
   0. sa_embs: 1xNx4096 [torch.float16]  # N = state_tokens + future_tokens + action_tokens
   1. vl_embs: 1x256x4096 [torch.float16]
   2. timesteps_tensor: 1 [torch.int64]
Outputs: 1
   0. output: 1xNx4096 [torch.float16]
=============================================

============= TRT Engine Detail =============
Engine file: gr00t_engine/action_decoder.engine
Inputs: 2
   0. model_output: 1xNx4096 [torch.float16]
   1. embodiment_id: 1 [torch.int64]
Outputs: 1
   0. output: 1xNx14 [torch.float16]  # N >= action_horizon
=============================================

# 4. 추론 실행 및 결과 출력
=== TensorRT Inference Results ===
action (4, 14)  # action_horizon=4, action_dim=14
```

### 로그 해석

1. **엔진 로드 단계**: 각 TensorRT 엔진이 로드될 때마다 입력/출력 텐서 정보가 출력됩니다.
   - `Inputs`: 엔진이 받는 입력 텐서들의 이름, shape, dtype
   - `Outputs`: 엔진이 출력하는 텐서들의 이름, shape, dtype

2. **Shape 정보**:
   - `pixel_values`: `[batch, channels, height, width]` = `[1, 3, 224, 224]`
   - `vit_embeds`: `[batch, num_patches, hidden_size]` = `[1, 196, 1152]`
   - `backbone_features`: `[batch, seq_len, hidden_size]` = `[1, 256, 4096]`
   - `actions`: `[batch, action_horizon, action_dim]` = `[1, 4, 14]`

3. **실행 순서**: 엔진들은 위에서 아래 순서로 로드되며, 실제 추론 시에는 다음 순서로 실행됩니다:
   - ViT → LLM → VL Self-Attention → State Encoder → [Action Encoder → DiT → Action Decoder] × 4 (denoising steps)

### 주의사항

- 배치 크기가 8을 초과하면 에러 발생 (엔진이 max_batch_size=8로 빌드됨)
- 엔진 파일이 없으면 `FileNotFoundError` 발생
- CUDA 메모리 부족 시 OOM 에러 발생 가능

