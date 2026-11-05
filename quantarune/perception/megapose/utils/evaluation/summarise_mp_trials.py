"""Summarise the MegaPose-guided ground-truth-off trials into one CSV.

Usage: python summarise_mp_trials.py <fixed_scene_runs_dir> <gazebo_data_dir> <prefix> <out.csv>
Pulls the hand-off source, the MegaPose checks, the scripted-insert outcome
and the official score for every session that starts with <prefix>.
"""
import csv
import glob
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def grab(lines, pattern, last=True):
    hits = [l for l in lines if re.search(pattern, l)]
    if not hits:
        return ""
    return hits[-1] if last else hits[0]


def main(runs_dir, data_dir, prefix, out_csv):
    rows = []
    for session in sorted(glob.glob(os.path.join(runs_dir, prefix + "*"))):
        name = os.path.basename(session)
        log = os.path.join(session, "logs", "policy.log")
        lines = [ANSI.sub("", l).rstrip() for l in open(log)] if os.path.exists(log) else []
        lines = [re.sub(r"^\[(INFO|WARN|ERROR)\] \[[0-9.]+\] \[aic_model\]: ", "", l) for l in lines]
        yaml = os.path.join(data_dir, name, "parquet_root", "experiment_1", "scoring.yaml")
        score, msg = "", ""
        if os.path.exists(yaml):
            y = open(yaml).read()
            m = re.search(r"^total:\s*(\S+)", y, re.M)
            score = m.group(1) if m else ""
            m = re.search(r"tier_3:\n\s*score:.*\n\s*message:\s*(.*)", y)
            msg = m.group(1).strip().strip('"') if m else ""
        handoff = grab(lines, r"^HANDOFF:")
        src = re.search(r"source=(\S+)", handoff)
        mp = grab(lines, r"megapose checks")
        mp_pose = grab(lines, r"LOOK\d+: megapose space_s")
        fin = grab(lines, r"^Scripted insert done")
        finish = grab(lines, r"^FINISH")
        looks = len([l for l in lines if re.match(r"LOOK\d+: template box", l)])
        rows.append({
            "session": name, "score_total": score, "tier3_message": msg,
            "template_looks": looks,
            "handoff_source": src.group(1) if src else "",
            "megapose_cam_depth_m": (re.search(r"cam_depth=([-\d.]+)", mp_pose) or [None, ""])[1] if mp_pose else "",
            "megapose_port_base": (re.search(r"port_base=(\[[^\]]*\])", mp_pose) or [None, ""])[1] if mp_pose else "",
            "megapose_checks": re.sub(r"^LOOK\d+: megapose checks ", "", mp) if mp else "",
            "insert_outcome": re.sub(r"^Scripted insert done\. ", "", fin) if fin else "",
            "finish": re.sub(r"^FINISH ", "", finish) if finish else "",
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["session"], "score", r["score_total"], "|", r["handoff_source"], "|", r["megapose_checks"], "|", r["insert_outcome"][:70])


if __name__ == "__main__":
    main(*sys.argv[1:5])
