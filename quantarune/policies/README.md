# policies

This folder holds the policies that we ran in the AIC simulator. A
policy reads the robot state and the cameras. It then sends motion
commands to the arm. 

## Layout

```
policies/
  act-fsm/                    the base ACT policy with four stages
  scripted-insert/            ACT approach, then a scripted insertion
  vision-gated/               ACT with vision features and a vision gate
  megapose-scripted-insert/   template and MegaPose approach, then a scripted insertion
```

## How the policies relate

| Policy | Extends |
|---|---|
| `RunACTStaged` | `RunACTCompetitor` (not in this repository) |
| `RunACTStagedScriptedInsert` | `RunACTStaged` |
| `RunACTStagedVisionGate` | `RunACTStaged` |
| `RunACTVision` | organiser `RunACT` |
| `RunMegaPoseScriptedInsert` | organiser `CheatCode` |

All ACT policies extend `RunACTStaged`. In the original repository all these 
files were in one ROS package. They import each other with relative imports.

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
