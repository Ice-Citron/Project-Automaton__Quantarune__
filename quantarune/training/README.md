# training

This folder holds the training code of Team Quantarune. There are two lines of 
work. `act/` is the ACT pipeline behind the stage policies in 
`../policies/act-fsm/`. On the other hand, `flow-matching/` is a 
flow-matching policy that we also built.

## Layout

```
training/
  act/
    build_sc_stage_act_datasets.py
    materialize_sc_stage_indexes_from_supervision.py
    lerobot_train_patched.py
    scripts/
      build_sc_stage_datasets.sh
      train_sc_stage_teachers.sh
      train_act_dataset.sh
    utils/
      lerobot_video_patch.py
    config/
      resolved_config.yaml
  flow-matching/
    config/  data/  models/  training/  eval/  utils/
  pyproject.toml
  pixi.toml
  .secrets.env.example
```

## act/

We trained one ACT head per stage of the SC insertion. The four stages
are `approach`, `align`, `guarded_insert` and `retry_or_finish`. The
finite-state machine in `../policies/act-fsm/RunACTStaged.py` switches
between these heads at run time.

| File | What it does |
|---|---|
| `materialize_sc_stage_indexes_from_supervision.py` | Writes one `stage_segment_index.csv` per stage from an SC stage supervision JSON |
| `build_sc_stage_act_datasets.py` | Builds four stage-specific LeRobot datasets with the centre, left and right cameras |
| `lerobot_train_patched.py` | Starts LeRobot training with a faster dataloader, without edits to site-packages |
| `utils/lerobot_video_patch.py` | Replaces LeRobot video frame decoding with a PyAV decoder |
| `scripts/build_sc_stage_datasets.sh` | Builds the SC stage manifest, then runs `build_sc_stage_act_datasets.py` |
| `scripts/train_sc_stage_teachers.sh` | Trains the four stage heads in sequence |
| `scripts/train_act_dataset.sh` | Trains one ACT policy on one dataset |
| `config/resolved_config.yaml` | The resolved config of the ACT run on `experiment_8` |

The order of one run is:

1. Record SC teleop episodes with `../data-collection/`.
2. Build the stage manifest and the four stage datasets.
3. Train the four stage heads.
4. Load the heads in `RunACTStaged` and evaluate in the Gazebo sim.

Checkpoints went to the Hugging Face repo `shng2025/aic-act-checkpoints`.
Each run got its own branch.

## flow-matching/

This is a flow-matching policy with a DiT (Diffusion Transformer)
denoiser. It also has a wrapper for LoRA fine-tuning of pi0.5. It uses
Hydra configs.

| Folder | What it holds |
|---|---|
| `config/` | Hydra configs for the model, data, training and sim evaluation |
| `data/` | LeRobot-to-training conversion, the PyTorch dataset, normalisation and action transforms |
| `models/` | The DiT denoiser, the ResNet-18 image encoder, the state encoder and the pi0.5 wrapper |
| `training/` | The training loop and the flow-matching loss |
| `eval/` | Automated sim evaluation through `aic_engine` and score parsing |
| `utils/` | 6D rotation helpers and the Hugging Face publisher |

> Note: No other code in this repository imports this folder. Our shipped 
> policies use ACT.

## Miscellaneous files

- `pyproject.toml` and `pixi.toml` record the environment we trained
  in. They still name the package `quantarune_training`.
- `.secrets.env.example` shows the two optional keys, `WANDB_API_KEY` and
  `HF_TOKEN`. Never commit a real `.secrets.env`.
- Some scripts still import `quantarune_training`. That package was split
  into `act/` and `flow-matching/`, and the vision code moved to
  `../perception/segmentation-model/model/`.
