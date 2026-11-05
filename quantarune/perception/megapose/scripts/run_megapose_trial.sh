#!/usr/bin/env bash
# One MegaPose-guided trial with ground truth off on a fixed scene config.
# Usage: run_megapose_trial.sh <session_name> <config.yaml>
set -uo pipefail
SP=/tmp/claude-1000/-mnt-sienarune-Black-Projects-Project-Automaton-Santa-Maria/e85be851-ca71-4636-bbd1-9a977f519d70/scratchpad
T=/mnt/sienarune/Black-Projects/Project-Automaton/Santa_Maria/code/aic-rangers-isaac-training
NAME=${1:?session name}
CFG=${2:?config yaml}
SESSION=$T/rangers_data/outputs/fixed_scene_runs/$NAME
mkdir -p "$SESSION/megapose"
export DBX_CONTAINER_MANAGER=docker GAZEBO_GUI=false LAUNCH_RVIZ=false
export AIC_GROUND_TRUTH=${AIC_GROUND_TRUTH:-false}
export PYTHONPATH=$SP/mp_rangers${PYTHONPATH:+:$PYTHONPATH}
export AIC_MEGAPOSE_LOG_DIR=$SESSION/megapose
echo "$(date +%H:%M:%S) start $NAME ground_truth=$AIC_GROUND_TRUTH policy mp_rangers.RunMegaPoseScriptedInsert config $CFG"
bash "$SP/pipe/run_engine_case_gtoff.sh" --policy-import mp_rangers.RunMegaPoseScriptedInsert \
  --config-file "$CFG" --session-name "$NAME" --result-timeout 420 > "$SESSION/runner.log" 2>&1
RC=$?
Y=$T/rangers_data/outputs/gazebo_data/$NAME/parquet_root/experiment_1/scoring.yaml
echo "$(date +%H:%M:%S) runner exit=$RC score=$(awk '/^total:/{print $2}' "$Y" 2>/dev/null) $(grep -E 'message:' "$Y" 2>/dev/null | tail -1 | cut -c1-120)"
echo "done"
