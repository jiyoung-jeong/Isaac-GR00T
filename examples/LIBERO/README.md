# LIBERO

Benchmark for studying knowledge transfer in lifelong robot learning. Includes multiple suites: **Spatial** (spatial reasoning), **Object** (object generalization), **Goal** (goal-conditioned learning), and **10 Long** (long-horizon multi-step tasks). Provides RGB images, proprioception data, and language task specifications.

For more information, see the [official website](https://libero-project.github.io/main.html).

---

# Jetson Thor quick start (Docker)

The commands below were originally tuned for multi-GPU dGPU servers (`NUM_GPUS=8`, `GLOBAL_BATCH_SIZE=640`).
For Jetson Thor, use single-GPU settings and launch inside the Thor container from `scripts/deployment/README.md`.

From repo root, build and run the Thor container:

```bash
cd docker && bash build.sh --profile=thor && cd ..

docker run --rm -it --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HOME}/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  gr00t-thor bash
```

If your shell shows `>` after pasting, it means the command is incomplete (usually due to a missing last line after `\`). Press `Ctrl+C` and re-run.

One-line equivalent:

```bash
docker run --rm -it --runtime nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host -v "$(pwd)":/workspace/repo -v "${HOME}/.cache/huggingface":/root/.cache/huggingface -w /workspace/repo -e HF_TOKEN="${HF_TOKEN:-}" -e WANDB_API_KEY="${WANDB_API_KEY:-}" gr00t-thor bash
```

Inside the container:

```bash
# (Optional) if you don't use Weights & Biases:
export USE_WANDB=0
# (Optional) extra safeguard to prevent any W&B prompt:
export WANDB_MODE=disabled

huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/

cp -r examples/LIBERO/modality.json examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/meta/

# Thor-friendly baseline:
# - single GPU (NUM_GPUS=1)
# - smaller global batch to fit memory
# NOTE: On Thor Docker, run `bash examples/finetune.sh` directly (do not wrap with `uv run`).
NUM_GPUS=1 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=32 SAVE_STEPS=1000 \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir outputs/libero_spatial_thor
```

What this run does:

- This is **fine-tuning (training)**, not immediate evaluation/inference.
- It starts from `nvidia/GR00T-N1.6-3B` and trains on the LIBERO Spatial dataset.
- Model checkpoints are written to `--output-dir` (for this example: `outputs/libero_spatial_thor`), typically at `SAVE_STEPS` intervals.
- During training, you should see step/loss/logging output in the terminal. If Weights & Biases is enabled, metrics are also sent there.

After training, use a saved checkpoint (for example `.../checkpoint-20000/`) for evaluation in the section below (`run_gr00t_server.py` + `rollout_policy.py`).

If you see:
`wandb: Enter your choice:`
it means W&B login is interactive.

- Choose **(2)** if you already have a W&B account and want online tracking.
- Choose **(1)** if you want to create a new account.
- Choose **(3)** to skip W&B visualization for this run.
- To avoid this prompt entirely on Thor, set `USE_WANDB=0` (and optionally `WANDB_MODE=disabled`) before starting training.

If you see a `flash-attn` build failure during `uv run ...` on Thor, you're likely using the wrong dependency stack (x86/root `pyproject.toml`) instead of the Thor image's preinstalled environment. Re-enter the container and run `bash examples/finetune.sh ...` directly as shown above.

If you still hit OOM on Thor, reduce `GLOBAL_BATCH_SIZE` further (e.g. `16` or `8`).

If you run Docker with `--rm`, avoid using `/tmp/...` for `--output-dir` because artifacts can disappear when the container exits. Prefer a mounted repo path such as `outputs/libero_spatial_thor`.

If your past training log says `Model saved to /tmp/libero_spatial_thor`, then:

- In the **same still-running container**, checkpoints are under `/tmp/libero_spatial_thor`.
- In a **new container session** (after exit with `--rm`), that `/tmp` path is gone.

Before exiting a container where training finished, copy artifacts to a mounted path:

```bash
mkdir -p outputs
cp -r /tmp/libero_spatial_thor outputs/libero_spatial_thor
```

### Q: Do I need to retrain to use TensorRT?

No. TensorRT is an **inference-time optimization** path.
For a trained checkpoint, you can run deployment/export steps (ONNX export + TensorRT engine build) and evaluate/infer without re-running finetuning.

### Thor TensorRT conversion (from deployment README)

Inside the `gr00t-thor` container, run:

```bash
# (Optional) choose latest persistent checkpoint under mounted repo path
LATEST_CKPT=$(ls -d outputs/libero_spatial_thor/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
echo "LATEST_CKPT=$LATEST_CKPT"

# Sanity checks before export (must exist as local directory)
test -d "$LATEST_CKPT"
test -f "$LATEST_CKPT/config.json"

# Any existing checkpoint directory works (not only checkpoint-20000).
# The numeric suffix is the training global step at save time
# (e.g., checkpoint-9000 = saved at step 9000).

# 1) Export ONNX from your trained model + LIBERO dataset shape
python scripts/deployment/export_onnx_n1d6.py \
  --model_path "$LATEST_CKPT" \
  --dataset_path examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/ \
  --embodiment_tag libero_panda \
  --output_dir outputs/libero_spatial_thor_onnx

# 2) Build TensorRT engine
python scripts/deployment/build_tensorrt_engine.py \
  --onnx outputs/libero_spatial_thor_onnx/dit_model.onnx \
  --engine outputs/libero_spatial_thor_onnx/dit_model_bf16.trt \
  --precision bf16 \
  --min-batch-size 1 \
  --opt-batch-size 4 \
  --max-batch-size 8

# Note:
# - `opt-batch-size` is an optimization-profile tuning point, not a guaranteed best-efficiency point.
# - Determine the real power/performance sweet spot by measurement (e.g., n_envs sweep: 1/2/4/5/8).

# 3) Run standalone inference with TensorRT
python scripts/deployment/standalone_inference_script.py \
  --model-path "$LATEST_CKPT" \
  --dataset-path examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/ \
  --embodiment-tag libero_panda \
  --traj-ids 0 \
  --inference-mode tensorrt \
  --trt-engine-path outputs/libero_spatial_thor_onnx/dit_model_bf16.trt \
  --denoising-steps 4
```

`run_gr00t_server.py` now supports TensorRT mode for sim eval server:

```bash
python gr00t/eval/run_gr00t_server.py \
    --model-path "$LATEST_CKPT" \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper \
    --inference-mode tensorrt \
    --trt-engine-path outputs/libero_spatial_thor_onnx/dit_model_bf16.trt
```

Use `--inference-mode pytorch` (default) to run without TensorRT.
For TensorRT sim-eval, make sure your engine batch profile covers your client `--n_envs`.
Example: with `--n_envs 5`, build engine with `--max-batch-size >= 5`. If your existing engine was built with batch 1 only, set `--n_envs 1` or rebuild the engine.
If you still see `ModuleNotFoundError: No module named 'scripts'`, your local `gr00t/eval/run_gr00t_server.py` is outdated. Update to the latest commit and re-run.
For deployment scripts that parse `EmbodimentTag` directly, use enum values like `libero_panda` (lowercase), not enum names like `LIBERO_PANDA`.
If `LATEST_CKPT` resolves to `/tmp/...` and export fails with Hugging Face repo-id validation, your checkpoint directory is likely missing in the current container session; use a persistent `outputs/...` checkpoint path.

---

# LIBERO evaluation benchmark result

> **Note:** The full task list is attached at the end of this document.

| Task      | Success rate       | max_steps | grad_accum_steps | batch_size |
|-----------|--------------------|-----------|------------------|------------|
| Spatial   | 195/200 (97.65%)        | 20K       | 1                | 640        |
| Goal      | 195/200 (97.5%)        | 20K       | 1                | 640        |
| Object    | 197/200 (98.45%)        | 20K       | 1                | 640        |
| 10 (Long) | 189/200 (94.35%)        | 20K       | 1                | 640        |

# Fine-tune LIBERO 10 (long)

To reproduce our finetune results, use the following commands to setup dataset and launch finetune experiments. Please remember to set `WANDB_API_KEY` since `--use-wandb` is turned on by default. If you don't have a WANDB account, please remove this argument:

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
```

Run the shared finetune launcher directly:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_10 \
    --state-dropout-prob 0.8
```

# Fine-tune LIBERO goal

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/meta/
## This is a patch for one of the episode where the image seems to be corrupted.
cp examples/LIBERO/patches/episode_000082.mp4 examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.wrist_image/
```

Run the shared finetune launcher directly:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_goal
```

# Fine-tune LIBERO object

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/meta/
```

Run the shared finetune launcher directly:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_object
```

# Fine-tune LIBERO spatial

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/meta/
```

Run the shared finetune launcher directly:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_spatial
```

# Evaluate checkpoint

First, setup the evaluation simulation environment. This only needs to run once for each simulation benchmark. After it's done, we only need to launch server and client.

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

Then, run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /tmp/libero_spatial/checkpoint-20000/ \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=720 \
    --env_name libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
    --n_action_steps 8 \
    --n_envs 5
```

## Thor-specific eval notes

If your training log says:
`Model saved to outputs/libero_spatial_thor`
use a **checkpoint subdirectory** as `--model-path` (for example `outputs/libero_spatial_thor/checkpoint-20000`).
Do not pass `outputs/libero_spatial_thor` directly, because `run_gr00t_server.py` expects a loadable model+processor directory.

Use two terminals inside the Thor container:

**Terminal 1 - Server (Thor):**
```bash
# Run from repo root first.
cd /workspace/repo

LATEST_CKPT=$(ls -d outputs/libero_spatial_thor/checkpoint-* | sort -V | tail -n 1)
echo "$LATEST_CKPT"
python gr00t/eval/run_gr00t_server.py \
    --model-path "$LATEST_CKPT" \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client (Thor):**
```bash
cd /workspace/repo

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=720 \
    --env_name libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
    --n_action_steps 8 \
    --n_envs 5
```

`gr00t/eval/rollout_policy.py` is the correct client script.
`gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python` is the Python interpreter created by `setup_libero.sh`.

Important: evaluate on the same LIBERO suite you fine-tuned on.

- If you trained on `libero_spatial_*`, use a Spatial task env (like the one above).
- If you evaluate a Spatial-trained checkpoint on a 10-Long/Goal/Object task (e.g., `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`), success can be near 0.

If `gr00t/eval/sim/LIBERO/libero_uv` does not exist, setup did not complete. Re-run:

```bash
bash gr00t/eval/sim/LIBERO/setup_libero.sh
ls gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
```

Tip: pick the latest checkpoint automatically:

```bash
LATEST_CKPT=$(ls -d outputs/libero_spatial_thor/checkpoint-* | sort -V | tail -n 1)
echo "$LATEST_CKPT"
python gr00t/eval/run_gr00t_server.py \
    --model-path "$LATEST_CKPT" \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

# Full task list

## Libero 10 (Long)
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
- `libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- `libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
- `libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- `libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
- `libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- `libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
- `libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`
- `libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`

## Libero Goal
- `libero_sim/open_the_middle_drawer_of_the_cabinet`
- `libero_sim/put_the_bowl_on_the_stove`
- `libero_sim/put_the_wine_bottle_on_top_of_the_cabinet`
- `libero_sim/open_the_top_drawer_and_put_the_bowl_inside`
- `libero_sim/put_the_bowl_on_top_of_the_cabinet`
- `libero_sim/push_the_plate_to_the_front_of_the_stove`
- `libero_sim/put_the_cream_cheese_in_the_bowl`
- `libero_sim/turn_on_the_stove`
- `libero_sim/put_the_bowl_on_the_plate`
- `libero_sim/put_the_wine_bottle_on_the_rack`

## Libero Object
- `libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_butter_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_milk_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket`

## Libero Spatial
- `libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
