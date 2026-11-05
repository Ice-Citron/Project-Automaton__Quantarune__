# data-collection

This folder holds the code that made our training data. All data comes
from the AIC simulator. The recorded datasets are not in this
repository. This folder is an archive, so the code does not run as is.

## Layout

```
data-collection/
  recorder/      ROS 2 package that records topics into LeRobot datasets
  gazebo-data/   scripted data runs in Gazebo with the CheatCode policy
  teleop/        launchers for human teleoperation in Gazebo and Isaac Sim
  replay/        policies that replay recorded trajectories
```

## How the parts fit

1. A run starts in Gazebo. A scripted policy, a teleoperator or a
   replay policy moves the arm.
2. `recorder/` records the robot state, the three cameras and the actions.
3. The recorder writes a LeRobot dataset in Parquet format.
4. A post-processing step adds the scores from the engine to each episode.
5. `../training/act/` builds the stage datasets from these episodes.

## recorder/

This is the ROS 2 package `aic_ros_recorder`.

| File | What it does |
|---|---|
| `recorder_node.py` | Records the AIC topics for one episode |
| `bag_to_parquet.py` | Converts a rosbag folder to LeRobot Parquet |
| `postprocess_scoring.py` | Adds the scores from `scoring.yaml` to a dataset |
| `recover_scoring_from_bags.py` | Makes a `scoring.yaml` again from the scoring bags |
| `read_dataset.py` | Opens a recorded dataset for inspection |
| `aic_metadata.py` | Shared metadata for the recorder |

## gazebo-data/

These scripts collect data with the CheatCode policy. CheatCode reads
the ground-truth port pose, so it gives clean demonstrations.

| File | What it does |
|---|---|
| `scripts/gazebo_data.py` | One launcher with the subcommands `cheatcode`, `record`, `intervention-policy` and `postprocess-scoring` |
| `scripts/generate_fixed_scene_batch.py` | Runs a batch of episodes on one fixed scene |
| `scripts/extract_engine_scoring.py` | Reads the final score block from an engine log |
| `bash/gazebo_*.sh` | Each one calls one subcommand of `gazebo_data.py` |
| `bash/generate_fixed_scene_batch.sh` | Calls `generate_fixed_scene_batch.py` |
| `bash/run_engine_case_cheatcode.sh` | Runs one engine case with CheatCode and extracts its score |

## teleop/

| File | What it does |
|---|---|
| `gazebo_teleop_launcher.py` | Starts Gazebo teleoperation with recording |
| `isaac_sim_teleop_launcher.py` | Starts teleoperation in Isaac Sim |
| `bash/gazebo_engine_teleop.sh` | Starts the engine for a manual teleop session |
| `bash/reset_gazebo_eval_stack.sh` | Resets the Gazebo stack between sessions |

## replay/

These policies replay a recorded successful SC trajectory. We used them
to make more demonstrations and to test the final insertion steps.

| File | What it does |
|---|---|
| `ReplaySCWaypoint.py` | Replays a recorded TCP trajectory without scoring TF |
| `ReplaySCAnchorGentleCommit.py` | Replays to a fixed anchor, then does a force-guarded commit |
| `ReplaySCVisualServo.py` | Replays the template, then centres the plug from the image |
| `ReplaySCVisualServoScriptedInsert.py` | Visual servo replay, then the scripted insert (v5) |
| `Run*.py` | Entry points that start the replay policies above |
| `replay_templates/` | The recorded trajectories that the replays read |

## Notes

- The scripts still use old paths such as `quantarune_data/`. These
  paths do not exist in this layout.
- The organiser code that these scripts call is in `../../submodules/aic`.
