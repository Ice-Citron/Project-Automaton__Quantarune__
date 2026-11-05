# aic_ros_recorder

ROS2 package that records AIC topics into a LeRobot-format dataset and, optionally, a separate rosbag capture.

## Installation & Setup

This package is run from the workspace root:

```bash
cd /home/enrique/Code/ws_aic/src
```

Install the Pixi environment once:

```bash
cd /home/enrique/Code/ws_aic/src
pixi install
```

After that, the normal runtime path is simply `pixi run ros2 run ...`. A manual
`colcon build` is not required just to launch the recorder.

If you changed Python code in this package and still see old behavior when using
`pixi run ros2 run ...`, refresh the Pixi-installed local package:

```bash
cd /home/enrique/Code/ws_aic/src
pixi reinstall ros-kilted-aic-ros-recorder
```

This is often needed because `pixi run` executes the package installed inside
`.pixi/envs/default/...`, which does not automatically track source edits.

## Usage

### Start the recorder node

```bash
cd /home/enrique/Code/ws_aic/src
pixi run ros2 run aic_ros_recorder aic_ros_recorder_node

# Auto-shutdown after N recorded trials
cd /home/enrique/Code/ws_aic/src
pixi run ros2 run aic_ros_recorder aic_ros_recorder_node -- --number-trials 10
```

If `--number-trials` is set, the recorder stops itself cleanly after the `stop`
message for the last configured episode. Pressing `Ctrl+C` also shuts the node
down gracefully and finalizes any active episode before exit.

### Start and stop recording

In another terminal:

```bash
# Start a new episode
pixi run ros2 topic pub --once /cheatcode/episode_status std_msgs/msg/String "{data: start}"

# Stop the current episode
pixi run ros2 topic pub --once /cheatcode/episode_status std_msgs/msg/String "{data: stop}"
```
### Postprocess scoring

After the run completes, place `scoring.yaml` inside the dataset root and run:

```bash
cd /home/enrique/Code/ws_aic/src
pixi run ros2 run aic_ros_recorder aic_ros_postprocess_scoring /tmp/aic_ros_recorder/experiment_1
```

This annotates the dataset in place by adding:

- `score_tier_1`
- `score_tier_2`
- `score_tier_3`

to the frame parquet files under `data/` and to the per-episode metadata parquet
files under `meta/episodes/`.

The recorder also stores the task in two parallel forms:

- numeric observation fields such as `observation.task.cable_type_id` and `observation.task.target_module_id`
- string observation fields such as `observation.task.command`, `observation.task.cable_type`, `observation.task.port_name`, and `observation.task.target_module_name`

### Inspect the parquet dataset with `read_dataset.py`

Use `read_dataset.py` when you want to quickly inspect what was written to the
LeRobot dataset without opening the parquet files manually.

Read one parquet file directly:

```bash
cd /home/enrique/Code/ws_aic/src
pixi run python aic_ros_recorder/src/aic_ros_recorder/read_dataset.py \
  --parquet_file /tmp/aic_ros_recorder/experiment_1/data/chunk-000/file-000.parquet
```

This prints:

- the parquet file path
- the list of columns
- the feature schema inferred from `meta/info.json`
- the last few rows stored in that parquet file

For newer datasets, the printed rows also expose the derived natural-language
command through `observation.task.command` and mirror it as `command` / `task`
in the expanded view when available.

Read the whole dataset through `LeRobotDataset`:

```bash
cd /home/enrique/Code/ws_aic/src
pixi run python aic_ros_recorder/src/aic_ros_recorder/read_dataset.py \
  --use_lerobot_dataset \
  --dataset_repo_id aic_ros_recorder/experiment_1 \
  --dataset_root /tmp/aic_ros_recorder/experiment_1
```

This prints:

- the dataset root
- the dataset repo id
- the full feature schema
- task metadata loaded from `meta/tasks.parquet`
- the last few rows across the dataset

After running the scoring postprocessing step, the same command will also show:

- `score_tier_1`
- `score_tier_2`
- `score_tier_3`

## Output

- **LeRobot dataset root**: `/tmp/aic_ros_recorder/experiment_<N>/`
- **Episode parquet files**: `/tmp/aic_ros_recorder/experiment_<N>/data/chunk-<CCC>/file-<FFF>.parquet`
- **Dataset metadata**: `/tmp/aic_ros_recorder/experiment_<N>/meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/chunk-<CCC>/file-<FFF>.parquet`
- **Per-camera videos**: `/tmp/aic_ros_recorder/experiment_<N>/videos/observation.images.<camera_name>/chunk-<CCC>/file-<FFF>.mp4`
- **Optional rosbags**: `/tmp/aic_ros_recorder_bags/experiment_<N>/episode_<M>/rosbag/`

Each `experiment_<N>` is a single LeRobot dataset. Every `start`/`stop` pair appends one new episode to that
dataset using LeRobot's native `data/`, `meta/`, and `videos/` layout.

The recorder intentionally stores exactly one `data/chunk-<CCC>/file-<FFF>.parquet`
per episode. This keeps the LeRobot directory structure intact while making each
trial easy to inspect and postprocess independently.

## Notes

- The `pixi run` prefix ensures consistent ROS environment even in multi-environment setups
- The recorder consumes `/cheatcode/task` and `/cheatcode/task_result` to store task metadata and the final action
  result as `task_result.success` and `task_result.message`
- The recorder is controlled only via `/cheatcode/episode_status` with `std_msgs/msg/String` values `start` and `stop`
- Camera streams are stored as LeRobot videos under `videos/observation.images.<camera_name>/...`
- Temporary per-frame PNGs may be created during encoding, but the recorder removes the `images/` cache after each
  episode so the final dataset keeps videos only
- The recorder intentionally opens and finalizes the LeRobot dataset once per episode so each trial gets its own
  parquet data file under `data/chunk-*/file-*.parquet`
- If `AIC_RECORDER_PARQUET_ROOT` is set, experiment datasets are created there
- If `AIC_RECORDER_ROSBAG_ROOT` is set, rosbags are created there; otherwise they default to `/tmp/aic_ros_recorder_bags`
- `scoring.yaml` is not consumed by the recorder at runtime; score enrichment is a separate postprocessing step
- `Error calling StartServiceByName for org.freedesktop.Notifications: Timeout was reached`
  can appear if desktop notifications are unavailable or slow; it does not mean
  the package build failed

## Development

### Refresh the installed package after editing code

For normal Python-side development:

```bash
cd /home/enrique/Code/ws_aic/src
pixi reinstall ros-kilted-aic-ros-recorder
```

If you changed dependency metadata or package configuration, refresh the full
Pixi environment instead:

```bash
cd /home/enrique/Code/ws_aic/src
pixi install
```

### Optional `colcon` workflow

`colcon` is optional for this package. You only need it if you specifically want
to work with the workspace `install/` tree or debug a ROS overlay workflow.

Run it from the workspace root:

```bash
cd /home/enrique/Code/ws_aic/src
pixi run colcon build --symlink-install --packages-select aic_ros_recorder
```

## Troubleshooting

### `ignoring unknown package 'aic_ros_recorder' in --packages-select`

You ran `colcon` from the wrong directory. Use the workspace root:

```bash
cd /home/enrique/Code/ws_aic/src
pixi run colcon build --symlink-install --packages-select aic_ros_recorder
```

Do not run that command from `/home/enrique/Code/ws_aic/src/aic`.

### Source code changed but `pixi run` still executes the old recorder

Reinstall the local Pixi package:

```bash
cd /home/enrique/Code/ws_aic/src
pixi reinstall ros-kilted-aic-ros-recorder
```

This is the right fix when stack traces point into:

- `.pixi/envs/default/lib/python.../site-packages/aic_ros_recorder/...`

and the file there still does not match your source tree.

### `pkg_resources.UnknownExtra: huggingface-hub ... has no such extra feature 'hf-transfer'`

If this still happens when starting the recorder, your installed recorder is stale
and still importing an older dependency chain. Refresh the Pixi environment:

```bash
cd /home/enrique/Code/ws_aic/src
pixi install
```

The recorder no longer depends on `lerobot_robot_aic` or its teleoperation
modules. If you still see a stack trace mentioning `lerobot_robot_aic`,
`lerobot_teleoperator_devices`, or `pygame`, you are not running the refreshed
recorder yet.

### `ImportError: cannot import name 'aggregate_pipeline_dataset_features'`

This means the recorder code does not match the installed LeRobot API version.
In this workspace, `lerobot==0.4.3` exposes `build_dataset_frame` but not the
older feature-construction helpers like `aggregate_pipeline_dataset_features`
and `create_initial_features`.

The recorder has been updated to build its dataset feature schema directly in the
format expected by `LeRobotDataset.create`, so after refreshing the installed package:

```bash
cd /home/enrique/Code/ws_aic/src
pixi reinstall ros-kilted-aic-ros-recorder
pixi run ros2 run aic_ros_recorder aic_ros_recorder_node
```

the node should start normally.

### `error: option --editable not recognized`

This only affects the optional `colcon` workflow. The normal Pixi runtime path
does not require `colcon`.

If you do want to use `colcon`, refresh the environment first:

```bash
cd /home/enrique/Code/ws_aic/src
pixi install
pixi run colcon build --symlink-install --packages-select aic_ros_recorder
```

### `Unable to connect to a Zenoh router`

Start a Zenoh router before launching the recorder if you are running the system
manually on the host:

```bash
cd /home/enrique/Code/ws_aic/src
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

If you are using the eval container via `/entrypoint.sh`, that setup already starts
the router for you.

### `ros2 bag command is not available in this environment`

This warning is not fatal. The recorder continues in parquet-only mode and still
writes the LeRobot dataset.

If you need rosbag capture as well, make sure your current Pixi environment
includes the `ros2 bag` CLI and plugins. The parquet dataset recording path does
not depend on rosbag.

### The recorder does not stop automatically after the evaluation run

Launch the recorder with a trial limit:

```bash
cd /home/enrique/Code/ws_aic/src
pixi run ros2 run aic_ros_recorder aic_ros_recorder_node -- --number-trials 10
```

The recorder will stop itself cleanly after the `stop` message of the final
episode. You can also always terminate it manually with `Ctrl+C`.
