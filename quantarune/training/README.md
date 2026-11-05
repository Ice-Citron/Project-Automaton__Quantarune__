# AIC Training

This is now the single training home for this branch.

Related data-generation and dataset-design notes now live in:

- `../quantarune_data/README.md`
- `../quantarune_data/bash/gazebo_cheatcode_session.sh`

Layout:

- `quantarune_training/scripts/`
  - Python training and conversion code
- `quantarune_training/bash/`
  - human-facing shell launchers
- `quantarune_training/outputs/`
  - converted datasets, checkpoints, and run artifacts

Data generation is intentionally separated:

- `../quantarune_data/`
  - cheatcode / teleop dataset planning, validation targets, and dataset-lane notes

Use these explicit entrypoints:

- `quantarune_training/scripts/prepare_act_multicam_dataset.py`
- `quantarune_training/scripts/dataset_peeper.py`
- `quantarune_training/scripts/dataset_viewer.py`
- `quantarune_training/scripts/audit_insertion_dataset.py`
- `quantarune_data/scripts/build_act_dataset_from_cases.py`
- `quantarune_data/scripts/publish_lerobot_dataset.py`
- `quantarune_training/bash/prepare_act_experiment_8_three_camera_dataset.sh`
- `quantarune_training/bash/peek_dataset.sh`
- `quantarune_training/bash/audit_dataset.sh`
- `quantarune_training/bash/run_act_experiment_8.sh`
- `quantarune_training/bash/run_act_experiment_8_smoke.sh`
- `quantarune_training/bash/run_act_generated_quick_smoke.sh`
- `quantarune_training/bash/build_act_dataset_from_cases.sh`
- `quantarune_training/bash/train_act_dataset.sh`
- `quantarune_training/bash/run_act_sfp_single.sh`
- `quantarune_training/bash/run_act_sfp_clean_compare.sh`
- `quantarune_data/bash/publish_generated_dataset.sh`
- `quantarune_training/bash/republish_act_runs.sh`
- `quantarune_training/bash/view_dataset.sh`
- `quantarune_training/bash/convert_experiment_8_for_flow_matching.sh`
- `quantarune_training/bash/run_flow_matching_experiment_8.sh`

## Secrets

Optional secrets live in:

- `quantarune_training/.secrets.env`

The bash launchers will source that file automatically when it exists.

Start from:

- `quantarune_training/.secrets.env.example`

Expected contents:

```bash
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Notes:

- `WANDB_API_KEY` is needed only if you want Weights & Biases logging.
- `HF_TOKEN` is needed only if you want Hugging Face uploads.
- The real `.secrets.env` file is gitignored already. The example file is the tracked template.

## Commands

From the training repo root:

```bash
cd aic-quantarune-isaac-training
```

ACT smoke test:

```bash
bash quantarune_training/bash/run_act_experiment_8_smoke.sh
```

Generated scripted-data ACT smoke test:

```bash
bash quantarune_training/bash/run_act_generated_quick_smoke.sh \
  --sc-batch-root /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_data/outputs/fixed_scene_batches/<batch_sc> \
  --sfp-batch-root /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_data/outputs/fixed_scene_batches/<batch_sfp>
```

Build one ACT dataset from one or more `cases.csv` manifests:

```bash
bash quantarune_training/bash/build_act_dataset_from_cases.sh \
  --run-name sfp_clean_198 \
  --cases-csv /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_data/outputs/fixed_scene_batches/sfp_200_04-20-02-53_sfp/cases.csv
```

Train one ACT run from an already-built dataset:

```bash
bash quantarune_training/bash/train_act_dataset.sh \
  --run-name sfp_clean_198 \
  --repo-id shng2025/aic-act-sfp-clean-198 \
  --dataset-root /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_training/outputs/act_datasets/sfp_clean_198_<timestamp>
```

SFP clean-only ACT comparison runs from prebuilt datasets:

```bash
bash quantarune_training/bash/run_act_sfp_clean_compare.sh \
  --dataset-root-clean-198 /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_training/outputs/act_datasets/sfp_clean_198_<timestamp> \
  --dataset-root-clean-198-plus-48 /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_training/outputs/act_datasets/sfp_clean_198_plus_48_<timestamp> \
  --repo-id-clean-198 shng2025/aic-act-sfp-clean-198 \
  --repo-id-clean-198-plus-48 shng2025/aic-act-sfp-clean-198-plus-48
```

Single SFP ACT run:

```bash
bash quantarune_training/bash/run_act_sfp_single.sh \
  --run-name sfp_clean_198 \
  --repo-id shng2025/aic-act-sfp-clean-198 \
  --cases-csv /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_data/outputs/fixed_scene_batches/sfp_200_04-20-02-53_sfp/cases.csv
```

Publish a merged scripted dataset to Hugging Face:

```bash
bash quantarune_data/bash/publish_generated_dataset.sh \
  --sc-batch-root /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_data/outputs/fixed_scene_batches/<batch_sc> \
  --sfp-batch-root /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_data/outputs/fixed_scene_batches/<batch_sfp> \
  --repo-id your-hf-user/10SC-10SFP-preAnnotation \
  --title "10SC + 10SFP preAnnotation"
```

For scripted smoke batches, keep Gazebo and RViz disabled during generation and inspect the exported previews afterward. Multi-case GUI runs are slower and can contaminate subsequent trials with stale visual state.

ACT 3-camera dataset prep only:

```bash
bash quantarune_training/bash/prepare_act_experiment_8_three_camera_dataset.sh
```

ACT 3-camera dataset prep from scratch:

```bash
AIC_FORCE_REBUILD_ACT_DATASET=1 \
  bash quantarune_training/bash/prepare_act_experiment_8_three_camera_dataset.sh
```

ACT main run:

```bash
bash quantarune_training/bash/run_act_experiment_8.sh
```

Republish the latest ACT smoke + main runs from `outputs/` to the Hub:

```bash
bash quantarune_training/bash/republish_act_runs.sh
```

Flow dataset conversion:

```bash
bash quantarune_training/bash/convert_experiment_8_for_flow_matching.sh
```

Flow main run:

```bash
bash quantarune_training/bash/run_flow_matching_experiment_8.sh
```

Dataset peeper:

```bash
bash quantarune_training/bash/peek_dataset.sh \
  /home/starforge-sf95/Black-Projects/Project-Automaton/Quantarune-Dataset/experiment_8
```

Another dataset example:

```bash
bash quantarune_training/bash/peek_dataset.sh \
  /home/starforge-sf95/Black-Projects/Project-Automaton/Quantarune-Dataset/record_10_trials_9Apr26 \
  --scale 1.0
```

Dataset audit:

```bash
bash quantarune_training/bash/audit_dataset.sh \
  /home/starforge-sf95/Black-Projects/Project-Automaton/Quantarune-Dataset/experiment_8
```

Another dataset audit example:

```bash
bash quantarune_training/bash/audit_dataset.sh \
  /home/starforge-sf95/Black-Projects/Project-Automaton/Quantarune-Dataset/record_10_trials_9Apr26
```

Peek the prepared 3-camera ACT dataset directly:

```bash
bash quantarune_training/bash/peek_dataset.sh \
  /home/starforge-sf95/Black-Projects/Project-Automaton/aic-quantarune-isaac-training/quantarune_training/outputs/act_datasets/experiment_8_three_camera_images \
  --scale 1.0
```

## Notes

- Python source stays in `scripts/`; shell launchers stay in `bash/`.
- outputs are written under `quantarune_training/outputs/`.
- W&B and Hugging Face behavior still come from the underlying training config and scripts.
- ACT training uses a patched launcher in `scripts/lerobot_train_patched.py` because the installed LeRobot package needs the PyAV video backend forced explicitly on this machine.
- The PyAV patch now caches decoded MP4 chunks per worker so repeated reads stop rescanning the same file from frame 0.
- Flow dataset conversion uses the same PyAV patch path for the same reason.
- ACT smoke runs write to `quantarune_training/outputs/act_experiment_8_smoke/<timestamp>/`.
- ACT main runs write to `quantarune_training/outputs/act_experiment_8/<timestamp>/`.
- ACT smoke runs publish to a per-run branch in `shng2025/aic-act-checkpoints` when `HF_TOKEN` is set, but they do not update `main`.
- ACT main runs publish to `shng2025/aic-act-checkpoints` when `HF_TOKEN` is set.
- `republish_act_runs.sh` republishes the latest saved smoke run and latest saved main run from `quantarune_training/outputs/` into that same repo.
- ACT optimization variants:
  - older 1-camera baseline: `quantarune_training/outputs/act_datasets/experiment_8_center_camera/`
  - current 3-camera path: `quantarune_training/outputs/act_datasets/experiment_8_three_camera_images/`
  - both ACT dataset-prep paths are pre-decoded image-backed datasets, not live MP4 decode during training
  - the pre-decoded optimization only affects offline training; simulator inference still uses live observations from Gazebo / Isaac, not dataset videos
- ACT Hub publishing uses the same strategy as flow:
  - each completed run uploads to its own branch
  - the same completed run is also synced to `main`
  - older per-run branches remain intact
- The ACT bash launchers auto-resolve the raw dataset at `Quantarune-Dataset/experiment_8`, then prepare a cheaper 3-camera image-backed ACT dataset from it automatically.
  Override with `RAW_DATASET_ROOT=/path/to/experiment_8` if the raw dataset lives elsewhere.
  Override with `DATASET_ROOT=/path/to/prepared_act_dataset` if you want to use a different already-prepared ACT dataset.
- The 3-camera prep launcher resumes from an existing partial build by default.
  Use `AIC_FORCE_REBUILD_ACT_DATASET=1` only when you intentionally want to delete the current prepared dataset and rebuild it from frame 0.
- The 3-camera prepared dataset lives at `quantarune_training/outputs/act_datasets/experiment_8_three_camera_images/`.
- The dedicated prep launcher is useful when you want to finish the image-backed dataset first and only start ACT training later.
- `build_act_dataset_from_cases.py` merges multiple successful per-case fixed-scene datasets into one local ACT-ready LeRobot dataset root.
- `build_act_dataset_from_cases.py` now supports bucket filtering so you can build clean-only ACT datasets from fixed-scene batch manifests.
- When `build_act_dataset_from_cases.py` runs with scene metadata encoding enabled, it appends structured case context to `observation.state`:
  - task mode one-hot: `sfp` vs `sc`
  - target NIC one-hot: `nic_card_mount_0` to `nic_card_mount_4`
  - NIC presence mask for each mount
  - NIC present-count one-hot
  - task-board pose values: `x`, `y`, `z`, `roll`, `pitch`, `yaw`
- This means ACT can condition on the fixed-scene case metadata without scraping `config.yaml` at training time; the merged dataset already carries the encoded context in its state vector.
- `run_act_generated_quick_smoke.sh` is the short-path for `10 SC + 10 SFP` scripted smoke batches:
  - it merges the successful case datasets from two batch roots
  - writes a merged local dataset under `quantarune_training/outputs/act_datasets/generated_scripted_smoke_<timestamp>/`
  - runs a quick ACT smoke train under `quantarune_training/outputs/act_generated_smoke/<timestamp>/`
- `run_act_sfp_clean_compare.sh` builds two clean-only SFP ACT datasets and trains them back-to-back:
  - run 1: clean `198`
  - run 2: clean `198 + 48`
  - each run publishes to its own Hugging Face model repo, writes checkpoints locally, uploads checkpoints to a timestamp branch, and syncs the completed run to `main`
- `run_act_sfp_single.sh` is the safer one-at-a-time path:
  - it is now only a convenience wrapper around the two explicit steps below
  - build with `build_act_dataset_from_cases.sh`
  - train with `train_act_dataset.sh`
- `build_act_dataset_from_cases.sh` is the explicit dataset-build step:
  - merges one or more fixed-scene `cases.csv` manifests into one ACT-ready local LeRobot dataset
  - filters by bucket (default `clean_success`)
  - appends scene metadata into `observation.state`
- `train_act_dataset.sh` is the explicit ACT-training step:
  - takes one already-built local dataset root
  - runs ACT training
  - publishes that run to a timestamp branch and `main` in the requested Hugging Face model repo
- `publish_generated_dataset.sh` merges SC + SFP batch roots into one local LeRobot dataset under `quantarune_data/outputs/hf_datasets/` and publishes it to a private Hugging Face dataset repo using `HF_TOKEN`.
- `peek_dataset.sh` is the preferred sanity-check entrypoint before training. It replays a dataset episode-by-episode and works against both raw local LeRobot datasets and prepared image-backed ACT datasets.
- `view_dataset.sh` remains as a compatibility alias for the same replay path.
- `audit_dataset.sh` is the preferred dataset quality entrypoint before deciding what to train on. It writes episode buckets, frame-level metrics, and a machine-readable summary under `quantarune_training/outputs/dataset_audit/<dataset_name>/`.
- On this machine the pixi OpenCV build has no GUI backend, so the peeper auto-falls back to exporting MP4 previews under `quantarune_training/outputs/dataset_viewer/<dataset_name>/`.
- The peeper now defaults to full-resolution export (`--scale 1.0`) so the fallback MP4s are easier to inspect visually.
- In that export fallback, the peeper now also tries to open the first generated MP4 automatically with `vlc` or `xdg-open`.
- Use `--no-open` if you only want the exported files without launching a player.
