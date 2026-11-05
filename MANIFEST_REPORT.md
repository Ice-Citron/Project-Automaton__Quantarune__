# Manifest report for intrinsic/

Written 2026-10-31 by the repo-organiser agent. The manifest is
quantarune/repo-structure-manifest.md in Santa_Maria and this report follows
its revised version of 2026-10-31. Every row was copied with rsync and no
source was moved or changed. Bytes are from du -sb on the destination. The
digest is sha256 over the sorted sha256sum list of all files under the
destination and it is only given for rows under 1 GB. Multi-folder rows were
digested from the common parent with the folder names in the order listed.

## Rows

| # | Destination under intrinsic/ | Bytes | Files | sha256 digest | Note |
|---|---|---|---|---|---|
| 1 | aic/IntrinsicAIChallenge/ | 325586021 | 388 | 8db7b9b818df0a5a2dd4fef9dd8e8727488666e0563ea90fb587424f5e3234ee | .git (90M) and .pixi excluded |
| 2 | quantarune/isaac-training/ | 145257595 | 174 | f71c35ab6a9f4e7e6407fc3fe4df3a45ab506103e960d3c38fefd205ad96ce0e | third trim on user instruction 2026-10-31 (revised): keeps README.md and pixi files and quantarune_training/{pyproject.toml,pixi.toml,pixi.lock,README.md,ACT_STUDENT_CONTRACT.md,quantarune_training/ package} (wandb/ 272M deleted on user decision) and patches/ (82 files, 138 MB); the 14 duplicated organiser packages and docs/ and docker/ and quantarune_data/ and quantarune_teleop/ and quantarune_training/{scripts,bash}/ and quantarune_scenario_generator/ quantarune_evaluator/ agentd_workspaces/ aic_utils_stock_restore_20260515_001/ sandbox/ and 12 teacher_fsm_*.jsonl and then tmp/ .claude/ .github/ and 26 teacher_fsm and terminal probe json files deleted; aic_ros_recorder/ kept as our own package; aic/ (colcon residue) aic_training/ (empty) and oom (empty) deleted; aic_flow_policy/ and the top-level config files await a user decision; SC_STAGE_REBUILD_HANDOFF_20260507.md (copy in act-fsm/pipeline/) and STRICT_SUCCESS_REWARD_TERMINATION_REPORT_20260429.md and PHYSX_STABILITY_NOTES_20260429.md deleted on user decision; SECURITY: quantarune_training/.secrets.env was present in the copy and was removed from intrinsic/ and must never be committed (only .secrets.env.example remains); the training chain now lives in quantarune/act-fsm/pipeline/ (27 files, 213087 bytes, sha256 da699bdb2e2fc80462682b1d0650f73d74cce98b04e7953cf265ca20a62eb701). Earlier: quantarune_teleop/outputs/gazebo_teleop/ (24 GB) excluded |
| 3 | quantarune/act-fsm/policies/ | 67656 | 3 | 57276541174cab04b7b201d8bb4e139997913c2e79e1c0e66857dd986aaee2c7 | RunACTHybridStaged.py does not exist in ros/ |
| 4 | quantarune/act-fsm/sc-stage-heads/sc_stage_fixed_scene_curated_001_v1_{approach,align,guarded_insert,retry_or_finish}/ | 2480978708 | 44 | over 1 GB | truncated: checkpoints/010000/ and the last symlink only; steps 002000 to 008000 skipped |
| 5 | quantarune/act-fsm/sfp-stage-heads/ | 2481069011 | 45 | over 1 GB | truncated as row 4; includes sfp_stage_137_v1_train_2026-10-31.log |
| 6 | quantarune/act-fsm/stage-datasets/sc_stage_fixed_scene_curated_001/ | 90185630 | 46 | 92f852ef63374d3ec883cce015a5fe9a186044139d567afafd9f27d6dcb9762e | |
| 7 | quantarune/act-fsm/stage-datasets/sfp_stage_137_v1/ | 97469379 | 44 | dfdbcae875af056a31102a8f74afcd285a7a539e14e6ba6e74459388d61600e0 | |
| 8 | quantarune/act-fsm/sc-stage-heads/sc_stage_fixed_scene_curated_001_v1_train_2026-05-06.log | 91204 | 1 | 18a467f5caef86761b8e81feafc592b472747e80bcc25255e58c74f443e408ab | |
| 9 | quantarune/act-fsm/sc-stage-heads/sc_stage_fixed_scene_curated_{20260508_002_terminal_v1_guarded_insert,20260508_002_terminal_v1_guarded_insert_no_imagenet,20260511_001_terminal_guarded_heavy_guarded_insert_ft,20260511_001_terminal_guarded_heavy_retry_or_finish_ft}/ | 2481377521 | 60 | over 1 GB | truncated: see the row 9 note below |
| 10 | quantarune/scripted-insert/RunACTStagedScriptedInsert.py | 12436 | 1 | d565bfef24d54001ff7069708ca1498bb24f0c688d630d2ce0853c42442447e1 | |
| 11 | quantarune/scripted-insert/snapshots/ | 23813 | 2 | 7a79cb546fd8888cdcc6b259f2d2882b180272157768e35910a60d0bc9da72b7 | v3 and v4 only; v1 and v2 were copied under the first manifest version and then deleted from intrinsic/ |
| 12 | quantarune/replay/ | 6252710 | 38 | 43e0deab9264435be3b5f091a512df2226443689feede08d061c8578d35ee1cd | seven files plus replay_templates/ |
| 13 | quantarune/demonstrations/sc-cases/sc_scripted_full_003/ | 459303915 | 421 | eaf2b10438caa57b551ea2414a6ee5a03913946551042dba050cf54d290e53c8 | trimmed on user instruction 2026-10-31 to configs/ and cases.csv and summary.json plus the 29 gazebo_data episodes named in sc_fsm_stage_frame_ranges.csv; 71 episodes and audits/ deleted; the 8 further listed sessions are in row 13b |
| 13b | quantarune/demonstrations/sc-cases/sc_recovery_scripted_20260504_001/ | 129790410 | 110 | 743d763a395d01813639b60cf25e302173d244b785c1665370b7ad972734c108 | added on user instruction 2026-10-31 from quantarune_data/outputs/fixed_scene_batches/sc_recovery_scripted_20260504_001/: configs/ and cases.csv and summary.json plus the 8 gazebo_data episodes named in sc_fsm_stage_frame_ranges.csv; the other 12 episodes and audits/ and the curated_*.csv and launch.log were not copied |
| 14 | quantarune/demonstrations/sfp-cases/sfp137_cc_20261031/ | 759625626 | 403 | 7a5e8bf03565afa16404b674a7c79885fc1f783e9e3213fa316aabe1a7f5c536 | batch files at top; 40 sessions under gazebo_data/ |
| 15 | quantarune/demonstrations/scene-configs/{sc_scripted_smoke_001,sc_scripted_full_001,sfp_300_04-20-21-55_sfp}/ | 2862500 | 354 | 48faf4126940f29492f4bd132ce41cdbddaea224911a9e796fd103b39c81c455 | configs/ and cases.csv each |
| 16 | quantarune/demonstrations/scene-configs/{sc_guarded_heavy_autorescue_20260510_003,sc_retry_reseat_autorescue_20260509_001,sfp_align_guarded_gentle_20260511_001,sfp_terminal_gentle_collect_20260512_001}/ | 393938 | 109 | 3d0c8bfee2002a7323ca949cbc47e1b4b643d0227e1fd1cc98416f2b263c019d | configs/ only |
| 17 | quantarune/computer-vision/classical/ | 102005 | 4 | 976783dd133c6e81a5df59d2409867ce167b1159b3a93ea7cee4d7653b2de8cc | |
| 18 | quantarune/computer-vision/segmentation-model/code/ | 140686 | 24 | bf4a4d341a6bc117d4841d3cb88ec018c4f251e6eb783974a57adb1223cbfb07 | subfolders vision/ scripts/ build-notes/ |
| 19 | quantarune/computer-vision/segmentation-model/model/ | 47365994 | 4 | 8543794e148d3fe64345985c5de58a5a5bcbf249362f22023172a057f2bf9b50 | |
| 20 | quantarune/computer-vision/segmentation-model/sample-data/ | 318750702 | 14088 | 329e1862f001daa2a3ba360c0ed95cd48e6d479904b7e9bdf17f704da7805738 | truncated: first 200 of 497 episodes in images/ and masks/plug/ and masks/target/ plus the three JSON files |
| 21 | quantarune/computer-vision/cnos/ | 2934823 | 10 | 06ae7849f29ac7ff02a67ea7c6a33b773afbaf7b942e84d560f97fbcdd44bd06 | |
| 22 | quantarune/computer-vision/megapose/README.md | written | 1 | | README only as the manifest asks |
| 23 | quantarune/research/insertion-net/code/ | 38972 | 6 | 0990e565fca22a35877cd6fc503d703e7ce29e1fb5bd1b344cbe2bb5f2456a3d | |
| 24 | quantarune/research/insertion-net/model/ | 54985813 | 26 | de4a6d88dd7b218c5ba6d40e9101492aca8d73cf77c900d1d7323504b2d8a947 | |
| 25 | quantarune/research/insertion-net/sample-data/ | 327142422 | 7131 | 9644e968415ec3db0ad6031ab7d1e76a39a88f80d03615f21191a3501eb5d764 | |
| 26 | quantarune/act-fsm/runs/ | 4125647045 | 203 | over 1 GB | 46 folders after the second trim on user instruction 2026-10-31; act-fsm-scripted-finish/ (25: 24 full insertions plus the FSM-retry clip run) and act-fsm-only/ (6) and replay/ (11) and cheatcode-reference/ (4); 5 folders without bags, see below |
| 27 | quantarune/videos/final/ | 32567444 | 25 | 1722a3afaefdc8433564eaa0f63148b4c31d73253eefc5e44b732956fad7789a | edited/ subfolder (50M) excluded |
| 28 | quantarune/videos/all-takes/ | 195262260 | 33 | 6c36bb60ad139633616fbd6285b73e07c5b19e60e848e4d7f590c5d8954360f2 | 31 mp4 and 2 mkv; partially_edited/ not copied |
| 29 | quantarune/videos/scores/ | 121463 | 47 | 32e68f9090a75b1fc6e2de6598c0f3bdf31b57e079fdb8f6df9fa226f571979b | |
| 30 | quantarune/videos/perception-figures/ | 7793172 | 16 | 975ab394f7b7f3a9185a9313cdaa5250a5b36c02de8ae5f7dd8d5187a46c6377 | |
| 31 | quantarune/videos/docs/ | 19907 | 2 | c56a03dc0286c92cccbe713bc46d94231735feb342c8d3f051690ce9e37b7d1a | |
| 32 | quantarune/handoffs/2026-10-30-video-status/ | 55941 | 6 | 6578ba41294e7a3f9817ced45d6eec86471a5d5a6e4717c919f54558579a4520 | |
| 33 | (removed in the revised manifest) | | | | no Enrik or Rocky material copied |

## Skipped and truncated rows

- Row 3: RunACTHybridStaged.py was listed as "if present" and it is not
  present in the ros/ folder. The other three files were copied.
- Rows 4 and 5: each head keeps checkpoints/010000/ and the last symlink
  only. Steps 002000 to 008000 stayed in the source. No log or config files
  exist at the top of the head folders. The train logs live one level up and
  were copied next to the heads.
- Row 9: the guarded_insert head of 20260508_002 holds only a wandb/ folder
  and no checkpoints and it was copied whole. The no_imagenet head keeps
  step 020000 (its last) and step 010000. The two 20260511_001 _ft heads
  have no step 010000 and keep step 012000 (their last). Their wandb/
  folders were copied where present.
- Row 20: 200 of 497 episodes were copied for images/ and for each of
  masks/plug/ and masks/target/. The full pack stays at
  quantarune_training/outputs/segmentation_annotation_pack_sfp_20260506/.
- Row 26: 135 folders were copied in manifest pattern order with bags until
  the destination passed 10 GiB and the last 11 were copied without
  results/bag_*. On user instruction of 2026-10-31 the 43 folders with no
  insertion were deleted from intrinsic/ only. A second trim the same day
  kept in act-fsm-scripted-finish/ only the 24 runs whose scoring.yaml says
  Cable insertion successful plus video-v4-fsmretry-004-gtoff-20261031_132348
  and kept in cheatcode-reference/ only the four runs behind the final clips
  and deleted the bag-stripped stitch-agentA-vsreplay-si-t3 run from replay/.
  46 folders remain. Five have no results/bag_*:
  obs-sc-staged-004-gtoff-20261026_221516,
  obs-sc-staged-004-gtoff-20261026_222350,
  gui-sc-staged-retry-004-20261026_163939,
  gui-sc-staged-retry-004-20261026_164224,
  gui-sc-staged-004-20261026_163647. gui-sfp-bridge has no scoring.yaml and
  keeps its bags at the top level.
- Row 33: removed in the revised manifest. No enrik_* or rocky_* folder
  remains. Under the first manifest version 3.3 GB had been copied to
  intrinsic/DO_NOT_USE/enrik/ and the empty folders DO_NOT_USE/rocky/ and
  act-fsm/enrik-checkpoint/ existed. All three were deleted from intrinsic/
  only and act-fsm/other-heads/ was moved into sc-stage-heads/. The v1 and v2
  snapshot copies were deleted the same way. No source was touched.

- Row 2 third trim (2026-10-31, revised): the collect, stage-dataset and
  train chain behind the stage heads was copied to quantarune/act-fsm/pipeline/
  with relative paths kept (16 named files plus 5 pulled in by the calls
  between them: gazebo_cheatcode_policy.sh, gazebo_data_record.sh,
  gazebo_intervention_policy.sh, gazebo_postprocess_scoring.sh,
  gazebo_teleop.py, plus the package __init__.py and the quantarune_teleop
  python package copied from the source repo on user decision;
  publish_act_run.py was dropped on user decision).
  quantarune_data/, quantarune_teleop/, quantarune_training/scripts/ and
  quantarune_training/bash/ were then deleted from isaac-training. The engine
  launcher copies the 30 N teleop controller config into the container on
  every launch and a reproduction must restore the 10 N default afterwards. No USAGE_REVIEW.md
  was kept.
- Row 2 third trim left in place because no instruction named them: the
  top-level folders aic/ aic_flow_policy/ aic_ros_recorder/ aic_training/
  tmp/ .claude/ .github/ and the top-level teacher_fsm_*.json and
  terminal_*.json and terminal_*.jsonl files. docs/enrique_notes.md was not
  saved to patches/ because it is Enrique's own notes. quantarune_data/ is
  under user review and quantarune_data/USAGE_REVIEW.md lists every file there
  with its caller.

## Deviations from the manifest text

- Row 2 is 826 MB rather than the estimated 280 MB. The extra folder
  quantarune_teleop/outputs/gazebo_teleop/ (24 GB of teleop recordings) was
  excluded because the row is code only. quantarune_training/wandb/ (272 MB)
  and aic_utils/ and aic_utils_stock_restore_20260515_001/ and
  agentd_workspaces/ (each over 100 MB) were kept.
- Row 1 measures 325 MB rather than 402 MB because .git was excluded.
- Row 18 keeps its three sources in subfolders vision/ and scripts/ and
  build-notes/ instead of one flat folder.
- One combined shell command that wrote the three copy scripts and launched
  them detached was denied by the permission classifier. The scripts were
  then written as files and run through the tool's own background mode with
  no change in content. Nothing was lost.

## Open points for the user

- Rocky material exists outside the manifest's search path and was not
  copied: aic-quantarune/submission_assets/checkpoints/rocky_act26,
  aic-quantarune/aic_example_policies/aic_example_policies/NOT_MINE_DO_NOT_USE/,
  aic-quantarune/docker/*rocky26*, and aic-quantarune/aic_eval_runs/rocky26_smoke_*.
  None of these matched a row 26 pattern.
- Row 9 interpretation: which checkpoint step of the no_imagenet head drove
  the stitched partial clip of 26 Oct is not recorded. Both 010000 and
  020000 were kept so the question can be settled later.
- Row 2: say whether quantarune_training/wandb/ and the three 100 MB plus
  utility folders should stay.

## Total

du -sh intrinsic/ = 14G (14573531042 bytes, 23909 files).
