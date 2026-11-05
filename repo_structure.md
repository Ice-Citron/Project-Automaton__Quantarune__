## Guidelines to the repo structure

Welcome to our submission for intrinsic's AI for Industry Challenge!!
Below is the structure of our repository.


```
intrinsic/
├── README.md
├── aic/                            this is the source repo cloned from the AI for Industry repo  
│   └── IntrinsicAIChallenge/
├── quantarune/
│   ├── README.md
│   ├── isaac-training/             our training repo, code only
│   │                               (the python package quantarune_training/ with the LeRobot video patch and the
│   │                                vision, models, training and eval modules the ACT heads were trained with;
│   │                                our ROS recorder package aic_ros_recorder; the small flow-policy package;
│   │                                patches/ holding our edits to the organiser packages; pixi files)
│   ├── act-fsm/
│   │   ├── policies/               RunACTStaged.py and some similar files 
│   │   ├── sc-stage-heads/         4 SC ACT heads + 3 later SC heads 
│   │   ├── sfp-stage-heads/        4 SFP ACT checkpoints 
│   │   ├── stage-datasets/         SC and SFP datasets, labelled by each stage 
│   │   └── runs/                   the scored trials, with their yaml scoring files per trial
│   ├── scripted-insert/
│   │   ├── RunACTStagedScriptedInsert.py (this is the best version we have yet!)
│   │   └── snapshots/              (we found v3 and v4 were good for some cases of sfp and sc)
│   ├── replay/                     visual-servo replay policy + template
│   │                               (RunSCVisualServoReplay plays back a recorded CheatCode demonstration for one scene
│   │                                and lets the centre camera correct the last approach; replay_templates/ holds the
│   │                                scene 026 recording; this produced the 91.6 full insertion on the four-card board)
│   ├── demonstrations/             (all of the teleoperated data we created, based off of roughly 200 different possible generated scenarios)
│   │   ├── sc-cases/
│   │   ├── sfp-cases/
│   │   └── scene-configs/
│   ├── computer-vision/            (some of our experimentations with computer vision in thsi project)
│   │   ├── classical/              vision_target_features.py, ReplaySCVisualServo.py
│   │   ├── segmentation-model/     (a trained U-Net 32-dim segmentation model)
│   │   │   ├── code/
│   │   │   ├── model/
│   │   │   └── sample-data/        (roughly 10,000 samples used for training the segmentation model!)
│   │   ├── cnos/                   (our plan for cnos + SAM and DINOv2 figures)
│   │   └── megapose/               (unsuccessful experimentation with megapose)
│   ├── research/
│   │   └── insertion-net/          (a venture based off of the InsertionNet paper from UWaterloo, which wasn't successful for us)
│   │       ├── code/
│   │       ├── model/
│   │       └── sample-data/        (4 hours worth of sample data collection for InsertionNet)
│   ├── videos/                     (clips we chose to display on the protfolio page/website)
│   │   ├── final/                  the clips chosen for editing
│   │   ├── all-takes/              every recording since 26 Oct
│   │   ├── scores/                 score JSON, compact JSON, phase timelines
│   │   ├── perception-figures/
│   │   └── docs/                   component README, narration script
│   └── handoffs/                   swarm handoff folder of 2026-10-30 (gitignore)
```
