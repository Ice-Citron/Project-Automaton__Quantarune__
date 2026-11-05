# policies

This folder holds the policies that we ran in the AIC simulator. A
policy reads the robot state and the cameras. It then sends motion
commands to the arm. This folder is an archive, so the code does not
run as is.

## Layout

```
policies/
  act-fsm/                    the base ACT policy with four stages
  scripted-insert/            ACT approach, then a scripted insertion
  vision-gated/               ACT with vision features and a vision gate
  megapose-scripted-insert/   template and MegaPose approach, then a scripted insertion
```

## How the policies relate

All ACT policies extend `RunACTStaged`. The MegaPose policy extends the
organiser's `CheatCode` policy instead.

| Policy | Extends |
|---|---|
| `RunACTStaged` | `RunACTCompetitor` (not in this repository) |
| `RunACTStagedScriptedInsert` | `RunACTStaged` |
| `RunACTStagedVisionGate` | `RunACTStaged` |
| `RunACTVision` | organiser `RunACT` |
| `RunMegaPoseScriptedInsert` | organiser `CheatCode` |

In the original repository all these files were in one ROS package.
They import each other with relative imports.

## act-fsm/

`RunACTStaged.py` is a finite-state machine with four stages:
`APPROACH`, `ALIGN`, `GUARDED_INSERT` and `RETRY_OR_FINISH`. Each stage
uses its own ACT head. `../training/act/` trained these heads.
`sfp_tuning_profiles.py` selects tuning values for the SFP task from an
environment variable.

## scripted-insert/

`RunACTStagedScriptedInsert.py` uses ACT for the approach and the
alignment. It then hands off to a scripted insertion that does not use
ground truth. The insertion descends slowly, wiggles, and retreats and
retries on contact. `archive/` holds the earlier versions v1, v3 and v4.

## vision-gated/

| File | What it does |
|---|---|
| `RunACTVision.py` | Adds compact vision features to the ACT state input |
| `RunACTStagedVisionGate.py` | Adds a vision supervision layer to `RunACTStaged` |

The vision features come from `../perception/classical/`.

## megapose-scripted-insert/

`RunMegaPoseScriptedInsert.py` runs with ground truth off. It finds the
SFP cage with template matching. It calls MegaPose once to check the
port pose. It then aligns the plug from the image and runs the scripted
insertion.

| Path | What it holds |
|---|---|
| `utils/template_port_detector.py` | The template-matching detector for the SFP cage |
| `config/templates_sfp137/` | The four cage templates from scene 137 |
| `config/calib_sfp_mount_to_port.json` | The calibration from the cage mesh to the port |
| `config/tip_in_tcp.json` | The plug tip pose in the gripper |

Three evidence trials with ground truth off inserted the cable on scene
137. They scored 87.21, 87.20 and 87.15. Template matching and the image
alignment drove the insertion. MegaPose was not accurate enough to move
the arm. `../perception/megapose/` records these trials.

## Notes

- `RunACTCompetitor.py`, `CheatCode.py` and `RunACT.py` are not in this
  folder. The organiser versions are in `../../submodules/aic`.
- `vision_target_features.py` is in `../perception/classical/`.
- The ACT checkpoints are not in this repository.
