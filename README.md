
# Background
Welcome to our submission for intrinsic's AI for Industry Challenge!! 

Intrinsic's AI for Industry Challenge is a competition run by Alphabet’s Intrinsic with NVIDIA and Foxconn. Teams are asked to create a robot policy that gets a Universal Robots UR5e robotic arm to insert a fiber-optic cable into a server port across two connector types (SC & SFP), inside Gazebo. There was a qualification round and a final round; we didn’t get past qualification, finishing 49th out of 196 teams. The competition scored every submission with Intrinsic’s custom-created evaluation-engine (the AIC engine) inside Gazebo, across three held-out trials. This is a very difficult challenge, because cable insertion is a hard task in robotics: the contact and friction dynamics of cable ports are incredibly difficult to model, and tends to be where policies fail.


## Project write-up and artefacts

Our write-up of the project is at
[sienarindustries.com/project/project-automaton](https://sienarindustries.com/project/project-automaton).

The Weights & Biases training report for the ACT policies (finished runs) is at
[docs/aic-act-training-wandb-report.pdf](docs/aic-act-training-wandb-report.pdf).

The MP4 files for our finalised videos, showcasing our policy and solution, are in `quantarune/videos/final/with_voice/` and the
scoring files for every clip are in `quantarune/videos/final/scores/`.

Watch the project on YouTube:

<table>
<tr><td width="50%" valign="top"><a href="https://www.youtube.com/watch?v=uIU_bVWS30E"><img src="https://img.youtube.com/vi/uIU_bVWS30E/hqdefault.jpg" width="100%"></a><br><sub><a href="https://www.youtube.com/watch?v=uIU_bVWS30E">Our Final Policy, an ACT-FSM doing cable-insertion</a> (<a href="https://www.youtube.com/watch?v=uIU_bVWS30E&t=3s">from 0:03</a>)</sub></td><td width="50%" valign="top"><a href="https://www.youtube.com/watch?v=havYEkYhJIg"><img src="https://img.youtube.com/vi/havYEkYhJIg/hqdefault.jpg" width="100%"></a><br><sub><a href="https://www.youtube.com/watch?v=havYEkYhJIg">Initial versions of our trained ACT model's poor performance in doing cable insertion</a></sub></td></tr>
</table>

<table>
<tr><td width="50%" valign="top"><a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/videos/final/with_voice/sc_easy_full_insertion_4_v3_retry_sc004_gtoff_20261027_121042_edited_voice.mp4"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/videos/final/with_voice/previews/sc_easy_full_insertion_4_v3_retry_sc004_gtoff_20261027_121042_edited_voice.gif" width="100%"></a><br><sub>Learned ACT stages plus the scripted finish on SC scene 004, ground truth off. Seats on the second attempt, score 96.86.</sub></td><td width="50%" valign="top"><a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/videos/final/with_voice/sfp_align_guarded_gentle_90point4_20260511_001_002_edited_voice.mp4"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/videos/final/with_voice/previews/sfp_align_guarded_gentle_90point4_20260511_001_002_edited_voice.gif" width="100%"></a><br><sub>CheatCode teacher on an SFP three-card board, ground truth on. Clean insertion, score 90.41.</sub></td></tr>
<tr><td width="50%" valign="top"><a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/videos/final/with_voice/sfp_terminal_gentle_collect_90point3_20260512_001_023_edited_voice.mp4"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/videos/final/with_voice/previews/sfp_terminal_gentle_collect_90point3_20260512_001_023_edited_voice.gif" width="100%"></a><br><sub>CheatCode teacher on an SFP five-card board, ground truth on. Clean insertion, score 90.59.</sub></td><td width="50%" valign="top"><a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/videos/final/with_voice/sc_guarded_heavy_autorescue_82point3_20260510_003_015_edited_voice.mp4"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/videos/final/with_voice/previews/sc_guarded_heavy_autorescue_82point3_20260510_003_015_edited_voice.gif" width="100%"></a><br><sub>CheatCode auto-rescue on an SC three-card board with a deliberately wrong target. Three rescues, score 82.35.</sub></td></tr>
<tr><td width="50%" valign="top"><a href="https://github.com/Ice-Citron/Project-Automaton__Quantarune__/blob/main/quantarune/videos/final/with_voice/sc_retry_reseat_autorescue_84point8_20260509_001_010_edited_voice.mp4"><img src="https://raw.githubusercontent.com/Ice-Citron/Project-Automaton__Quantarune__/main/quantarune/videos/final/with_voice/previews/sc_retry_reseat_autorescue_84point8_20260509_001_010_edited_voice.gif" width="100%"></a><br><sub>CheatCode auto-rescue on an SC one-card board with a larger wrong target. Two rescues, score 84.84.</sub></td></tr>
</table>

## Understanding our results 

NOTE: it is very difficult and infeasible to include every obtained result in this table, so we decided to include results which showcase the most variety across scenarios and scores too.

| Scenario | Method | Max score | Notes |
|---|---|---|---|
| SC scene 004, no cards, ground truth off | learned stages + scripted finish | 98.00 | 35 full insertions over the stitch batches, 7 of 10 on the v4 and v5 batches |
| SC scene 004, ground truth off | learned stages + scripted finish, one retry | 96.86 | the opening clip |
| SC scene 004, ground truth off | learned stages + scripted finish, FSM retry twice then finish | 53.73 | the retry clip, ends a centimetre short |
| SC scene 005, two cards, ground truth off | learned stages + scripted finish | 40.62 | lands about 2 cm beside the socket, no insertion |
| SC scene 026, four cards, ground truth off | visual-servo replay | 91.57 | one full insertion, eight partials at 53 to 54 |
| SFP scene 137, four cards, ground truth off | learned stages | 37.8 to 42.4 | four attempts, all about 5 cm off |
| SFP three-card and five-card boards, ground truth off | ACT FSM | 90.41 and 90.59 | reference runs, not our policy |
| SC carded boards, ground truth on | ACT FSM | 82.35 and 84.84 | reference runs showing back-out and re-centre |
| SFP scene 137, ground truth off, after the challenge | taught templates + image servo + scripted insert | 87.21, 87.20, 87.15 | three of three inserted, see MegaPose below |

In the qualification round our submission finished 49th of 196 teams. The
scored bags behind every row are in `quantarune/act-fsm/runs/` and on
Hugging Face under `quantarune-act-fsm-scored-trials`.

## Approach

### Four learned stages and a state machine

`RunACTStaged.py` wraps four ACT policies in a small state machine with the
states APPROACH, ALIGN, GUARDED_INSERT and RETRY_OR_FINISH. Each state has
its own head trained on the slice of the demonstrations that belongs to
that phase, and the wrapper decides which head runs and when to switch. 

The heads see the three wrist cameras at 256 by 288, the 48-value robot state
and nothing else, and each predicts a chunk of 100 actions of 19 values.
APPROACH flies the plug from the start pose to a hover point above the
target, ALIGN squares the tip up over the socket and waits until the
predicted pose stops moving, and GUARDED_INSERT pushes down while the
wrapper watches the force sensor and the descent. If the plug has not
descended enough within the guarded-insert window the wrapper backs out
about 10 mm, settles, and hands control back to ALIGN for another attempt.
After the retry budget is spent it hands over to the scripted finish. 

Every transition is a rule in the wrapper rather than a prediction, which makes a
run inspectable after the fact and gives the retry a fixed place to live.
The heads were trained for 10,000 steps at batch size 8 with the LeRobot
ACT trainer and the video patch in `isaac-training/`; the exact collect,
dataset and train scripts are in `act-fsm/pipeline/`.

### Scripted finish

The learned guarded insert reliably brings the plug onto the port axis but
stalls about 13 mm above the seat, so `RunACTStagedScriptedInsert.py` takes
over at that point. 

It descends straight down in the base frame at a slow
fixed rate, adds a small circular wiggle whose radius only grows while
depth progress stalls, and reads no port pose at any time. A rolling stall
test triggers a retry: back off, settle, and descend again, and from v4 each
retry re-descends from a hand-off position shifted by a small offset in a
new direction, since retries at the same spot met the same lip. v5 widens
those offsets by one step per cycle. 

The finish is compliant across the insertion axis and firm along it, and the force cap of the controller ends
the attempt before the 20 N penalty (better than getting the penalty and risking the score for it). We have included some versions behind the clips in
`scripted-insert/snapshots/` and the
logs showed the outcome was mostly fixed at hand-off, which is why the scripted insert worked so well. 

### Teacher and demonstrations

Every demonstration within the dataset was recorded by CheatCode, a scripted teacher that
reads the true port pose from the simulator and inserts the cable itself.
To keep the demonstrations realistic it adds a small deliberate error to
the target, typically one to three millimetres and a degree or two of
yaw, which fades to zero over the last 10 mm of the descent. It has three
modes: full, where it does the whole insertion; handoff, where it stops
above the port; and auto_rescue, where it descends on a deliberately wrong
target, misses, backs out, re-centres on the true pose and tries again,
up to three times. The SC training set is `sc_scripted_full_003` (100
generated cases, of which the 29 clean episodes in the stage frame ranges
were used) plus the 8 recovery episodes of `sc_recovery_scripted_20260504_001`,
and the SFP set is the 40 sessions of `sfp137_cc_20261031`. Scenes were
drawn from generated configs with a qualification-like mix of network
cards, listed under `demonstrations/scene-configs/`. Each episode was cut
into the four stages by a reference-pose heuristic with a minimum segment
of 8 frames, giving the stage datasets on Hugging Face (SC: 37 episodes
per stage, SFP: 40 per stage).


## What worked and what did not

On the training board the full pipeline seats the plug in about seven of
ten runs with scores near 98, and the retry state does what it was built
for: when guarded insert makes no progress the wrapper backs out, re-aligns
and tries again. 

On boards with network cards the same weights reach the
right area but miss the socket by one to three centimetres, because the
learned stages carry no fine port localisation - this is the gap we identified, but due to time constraints of the competition, we couldn't cater for fine port localisation. 

From the replay you can see it seated the plug on the
four-card board but only for that scene. 

We ran perception experiments
(segmentation, SAM, CNOS-style selection, MegaPose) to find the port in
the image but unfortunately none of them reached the millimetre accuracy that was needed. What would help is a port locator which is accurate to a millimetre feeding the ALIGN stage.

## How to run

The policies run inside the organisers' (Intrinsic, NVIDIA) evaluation stack, which is the
Gazebo and ROS 2 setup of `aic/IntrinsicAIChallenge/` with the
`aic_model` policy interface. The steps below name the scripts; the paths
inside them still point at the original repository layout and its pixi
environment, so expect to adjust them.

1. Set up the organisers' stack from `aic/IntrinsicAIChallenge/` and apply
   our edits from `quantarune/isaac-training/patches/`, which hold every
   file we changed in the organiser packages at its relative path.
2. Record demonstrations with the CheatCode teacher through
   `quantarune/act-fsm/pipeline/.../run_engine_case_cheatcode.sh`. The
   engine launcher copies a controller config with a 30 N wrench cap into
   the container; restore the organiser default of 10 N afterwards.
3. Build the stage datasets with `build_sc_stage_datasets.sh`, which calls
   `build_sc_stage_act_datasets.py`, or download them from Hugging Face.
4. Train the four heads with `train_sc_stage_teachers.sh`, which calls
   `train_act_dataset.sh` and `lerobot_train_patched.py`.
5. Run a trial with `RunACTStagedScriptedInsert.py` as the `aic_model`
   policy. The stage heads are selected with the environment variables
   `AIC_ACT_STAGE_ROUTE_APPROACH`, `_ALIGN`, `_GUARDED_INSERT` and
   `_RETRY_OR_FINISH`, and the scored bag and its scoring file land in the
   engine's results folder.

## Data and models on Hugging Face

| Hugging Face repo | Type | Contents |
|---|---|---|
| [quantarune-act-fsm-sc-demonstrations](https://huggingface.co/datasets/shng2025/quantarune-act-fsm-sc-demonstrations) | dataset | SC teacher demonstrations |
| [quantarune-act-fsm-sfp-demonstrations](https://huggingface.co/datasets/shng2025/quantarune-act-fsm-sfp-demonstrations) | dataset | SFP teacher demonstrations |
| [quantarune-act-fsm-sc-stage-dataset](https://huggingface.co/datasets/shng2025/quantarune-act-fsm-sc-stage-dataset) | dataset | SC LeRobot dataset cut into the four stages |
| [quantarune-act-fsm-sfp-stage-dataset](https://huggingface.co/datasets/shng2025/quantarune-act-fsm-sfp-stage-dataset) | dataset | SFP LeRobot dataset cut into the four stages |
| [quantarune-act-fsm-scored-trials](https://huggingface.co/datasets/shng2025/quantarune-act-fsm-scored-trials) | dataset | scored trial bags with their scoring files |
| [quantarune-act-fsm-sc-scripted-insert](https://huggingface.co/shng2025/quantarune-act-fsm-sc-scripted-insert) | model | scripted insert model |

You may notice that files over 50 MB are not in this git repository and you would be correct. Each affected folder has a
`LARGE_FILES_NOT_IN_GIT.md` that lists them and says where they are located (all on Hugging Face).


## Acknowledgements

The scoring engine, running and simulation of the AI for Industry Challenge is done by Intrinsic, NVIDIA and Foxconn, as well as all related assets (thanks!). 
Action Chunking Transformer usage came from the original authors of
ACT (Zhao et al., 2023) 
Also to thank are the authors of Segment Anything, DINOv2, CNOS,
MegaPose and InsertionNet, whose work the perception experiments follow.

## Licence

MIT, see [LICENSE](LICENSE).
