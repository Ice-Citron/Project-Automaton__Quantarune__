"""Compare MegaPose poses from the Space with the ground-truth port pose.

Usage: python compare_to_gt.py <results.json> <out_csv>
results.json rows carry pose_T_cam_obj (MegaPose) and T_cam_port (bag GT).
The mesh origin and the port link origin are not the same point, so the
translation error is reported raw and after removing the median offset
expressed in the GT port frame.
"""
import csv
import json
import sys

import numpy as np


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main(results_json, out_csv):
    res = json.load(open(results_json))["results"]
    rows = []
    offsets = []
    for r in res:
        if not r.get("ok") or r.get("pose_T_cam_obj") is None:
            rows.append([r["file"], False] + [None] * 7)
            continue
        Te = np.array(r["pose_T_cam_obj"])
        Tg = np.array(r.get("T_cam_mesh_gt") or r["T_cam_port"])
        dt = Te[:3, 3] - Tg[:3, 3]
        off_port = Tg[:3, :3].T @ dt
        offsets.append(off_port)
        rows.append([r["file"], True, round(float(Tg[2, 3]), 4), round(float(Te[2, 3]), 4),
                     round(float(np.linalg.norm(dt)), 4), round(float(dt[2]), 4),
                     round(float(np.linalg.norm(dt[:2])), 4),
                     round(rot_angle_deg(Te[:3, :3], Tg[:3, :3]), 2), r["seconds"]])
    med = np.median(np.array(offsets), axis=0) if offsets else np.zeros(3)
    for row, r in zip(rows, res):
        if row[1]:
            Te = np.array(r["pose_T_cam_obj"]); Tg = np.array(r.get("T_cam_mesh_gt") or r["T_cam_port"])
            resid = Te[:3, 3] - (Tg[:3, 3] + Tg[:3, :3] @ med)
            row.append(round(float(np.linalg.norm(resid)), 4))
        else:
            row.append(None)
    header = ["file", "ok", "gt_depth_m", "est_depth_m", "trans_err_m", "depth_err_m",
              "lateral_err_m", "rot_err_deg", "seconds", "trans_err_after_median_offset_m"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    ok = [r for r in rows if r[1]]
    print("frames ok:", len(ok), "/", len(rows))
    print("median offset of mesh origin in port frame (m):", np.round(med, 4).tolist())
    for name, i in (("trans_err_m", 4), ("depth_err_m", 5), ("lateral_err_m", 6), ("rot_err_deg", 7),
                    ("trans_err_after_median_offset_m", 9)):
        v = np.array([r[i] for r in ok], dtype=float)
        print(f"{name:34s} median {np.median(v):.4f}  mean {v.mean():.4f}  max {v.max():.4f}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
