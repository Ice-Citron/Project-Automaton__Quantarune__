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