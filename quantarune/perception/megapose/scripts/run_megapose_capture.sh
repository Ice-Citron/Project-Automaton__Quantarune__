#!/usr/bin/env bash
# One CheatCode reference episode on SFP scene 137 (ground truth on) with
# full-resolution centre-camera frames saved every 0.5 s for the MegaPose test.
set -uo pipefail
SP=/tmp/claude-1000/-mnt-sienarune-Black-Projects-Project-Automaton-Santa-Maria/e85be851-ca71-4636-bbd1-9a977f519d70/scratchpad
T=/mnt/sienarune/Black-Projects/Project-Automaton/Santa_Maria/code/aic-rangers-isaac-training
CFG=$T/rangers_data/outputs/fixed_scene_batches/sfp_300_04-20-21-55_sfp/configs/sfp_300_04-20-21-55_sfp_137.yaml
NAME=${1:?session name}
SESSION=$T/rangers_data/outputs/fixed_scene_runs/$NAME
FRAMES=$SESSION/center_camera_frames
mkdir -p "$SESSION" "$FRAMES"
cp "$SP/pipe/save_center_camera_frames.py" "$SESSION/"
export DBX_CONTAINER_MANAGER=docker GAZEBO_GUI=false LAUNCH_RVIZ=false
export PYTHONPATH=$SP/cc_rangers${PYTHONPATH:+:$PYTHONPATH}
echo "$(date +%H:%M:%S) start $NAME (ground_truth:=true, policy cc_rangers.CheatCode, config $CFG)"
bash "$SP/pipe/run_engine_case_cheatcode.sh" --policy-import cc_rangers.CheatCode \
  --config-file "$CFG" --session-name "$NAME" > "$SESSION/runner.log" 2>&1 &
RUNNER=$!
sleep 20
echo "$(date +%H:%M:%S) start frame saver"
distrobox enter aic_eval -- bash -lc "source /opt/ros/kilted/setup.bash && /usr/bin/python3 $SESSION/save_center_camera_frames.py $FRAMES 0.5" > "$SESSION/saver.log" 2>&1 &
SAVER=$!
wait $RUNNER
RC=$?
echo "$(date +%H:%M:%S) runner exit=$RC"
docker exec aic_eval pkill -INT -f save_center_camera_frames >/dev/null 2>&1 || true
sleep 3
kill $SAVER >/dev/null 2>&1 || true
echo "frames saved: $(ls "$FRAMES"/*.ppm 2>/dev/null | wc -l)"
Y=$T/rangers_data/outputs/gazebo_data/$NAME/parquet_root/experiment_1/scoring.yaml
echo "score: $(awk '/^total:/{print $2}' "$Y" 2>/dev/null)  $(grep -E 'message:' "$Y" 2>/dev/null | tail -1 | cut -c1-120)"
echo "$(date +%H:%M:%S) done"
