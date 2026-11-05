#!/usr/bin/env bash
# Three ground-truth-off evidence trials of the final configuration.
export AIC_GROUND_TRUTH=false AIC_MEGAPOSE_TEMPLATE_YAW_SEARCH_DEG=0 AIC_MEGAPOSE_HANDOFF_BIAS_XY_MM=0,2.0
export AIC_SCRIPTED_INSERT_STIFFNESS=300,300,500,85,85,85 AIC_SCRIPTED_INSERT_STOP_FORCE_N=19 AIC_SCRIPTED_INSERT_DEPTH_M=0.08
export AIC_SCRIPTED_INSERT_STALL_WINDOW_S=3.0 AIC_SCRIPTED_INSERT_STALL_PROGRESS_M=0.0002
export AIC_MEGAPOSE_USE=none AIC_MEGAPOSE_YAW_SOURCE=template AIC_SCRIPTED_INSERT_RETRIES=8 AIC_SCRIPTED_INSERT_RETRY_OFFSET_M=0.0015
export AIC_MEGAPOSE_ALIGN_PLUG_AXIS=1
for n in 019 020 021; do
  /tmp/claude-1000/-mnt-sienarune-Black-Projects-Project-Automaton-Santa-Maria/e85be851-ca71-4636-bbd1-9a977f519d70/scratchpad/pipe/run_megapose_trial.sh sfp137_megapose_gtoff_20261104_$n /mnt/sienarune/Black-Projects/Project-Automaton/Santa_Maria/code/aic-rangers-isaac-training/rangers_data/outputs/fixed_scene_batches/sfp_300_04-20-21-55_sfp/configs/sfp_300_04-20-21-55_sfp_137.yaml
  sleep 5
done
echo "chain done"
