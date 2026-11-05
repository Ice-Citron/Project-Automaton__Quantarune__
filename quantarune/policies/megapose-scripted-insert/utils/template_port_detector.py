"""Template-matching detector for the SFP cage in the centre camera.

Templates are crops of the cage taught once from reference frames with a
known box. Detection runs normalised cross-correlation on grey images over
a small scale pyramid and returns the best box and score.
"""
import json
import os

import cv2
import numpy as np

SCALES = tuple(round(0.6 + 0.04 * i, 2) for i in range(24))   # 0.60 .. 1.52 in 4 % steps


def _grey(rgb):
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0)


class TemplatePortDetector:

    def __init__(self, bank_dir):
        spec = json.load(open(os.path.join(bank_dir, "templates.json")))
        self.templates = []
        self.depth_m = {}
        self.centre_offset_px = {}
        for t in spec["templates"]:
            im = cv2.imread(os.path.join(bank_dir, t["file"]), cv2.IMREAD_GRAYSCALE)
            self.templates.append((t["file"], cv2.GaussianBlur(im, (3, 3), 0)))
            self.depth_m[t["file"]] = float(t["depth_m"])
            self.centre_offset_px[t["file"]] = t.get("centre_minus_port_px", [0.0, 0.0])

    def detect(self, rgb, min_score=0.55):
        g = _grey(rgb)
        H, W = g.shape
        best = None
        for name, tpl in self.templates:
            for s in SCALES:
                th, tw = int(tpl.shape[0] * s), int(tpl.shape[1] * s)
                if th < 16 or tw < 16 or th >= H or tw >= W:
                    continue
                t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if best is None or val > best[0]:
                    best = (float(val), [loc[0], loc[1], loc[0] + tw, loc[1] + th], name, s)
        if best is None or best[0] < min_score:
            return None, best[0] if best else 0.0
        return best[1], best[0]

    def detect_with_range(self, rgb, min_score=0.55, angles_deg=(0.0,)):
        """Box, score, camera range from the template scale and the in-plane
        rotation (deg) of the match relative to the taught templates."""
        g0 = _grey(rgb)
        H, W = g0.shape
        best = None
        for ang in angles_deg:
            if ang == 0.0:
                g = g0
            else:
                M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
                g = cv2.warpAffine(g0, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for name, tpl in self.templates:
                for s in SCALES:
                    th, tw = int(tpl.shape[0] * s), int(tpl.shape[1] * s)
                    if th < 16 or tw < 16 or th >= H or tw >= W:
                        continue
                    t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
                    _, val, _, loc = cv2.minMaxLoc(res)
                    if best is None or val > best[0]:
                        best = (float(val), [loc[0], loc[1], loc[0] + tw, loc[1] + th], name, s, ang)
        if best is None or best[0] < min_score:
            return None
        val, box, name, s, ang = best
        # refine the scale with a parabola through the scores at s-step, s, s+step
        if ang == 0.0:
            tpl = dict(self.templates)[name]
            scores = {}
            for ss in (round(s - 0.04, 2), s, round(s + 0.04, 2)):
                th, tw = int(tpl.shape[0] * ss), int(tpl.shape[1] * ss)
                if th < 16 or tw < 16 or th >= H or tw >= W:
                    scores = {}
                    break
                res = cv2.matchTemplate(g0, cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
                scores[ss] = float(cv2.minMaxLoc(res)[1])
            if len(scores) == 3:
                y0, y1, y2 = [scores[k] for k in sorted(scores)]
                denom = y0 - 2 * y1 + y2
                if denom < 0:
                    s = s + 0.04 * 0.5 * (y0 - y2) / denom
        if ang != 0.0:
            # map the box centre back into the unrotated image
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -ang, 1.0)
            ux, uy = (M @ np.array([cx, cy, 1.0]))
            bw, bh = box[2] - box[0], box[3] - box[1]
            box = [ux - bw / 2.0, uy - bh / 2.0, ux + bw / 2.0, uy + bh / 2.0]
        off = self.centre_offset_px[name]
        port_px = [(box[0] + box[2]) / 2.0 - off[0] * s, (box[1] + box[3]) / 2.0 - off[1] * s]
        return {"box": box, "score": val, "template": name, "scale": s,
                "range_m": self.depth_m[name] / s, "rotation_deg": float(ang), "port_px": port_px}


    def detect_rotation(self, rgb, template, scale, angles_deg):
        """In-plane rotation (deg) of the match with the template and scale fixed."""
        tpl = dict(self.templates)[template]
        g0 = _grey(rgb)
        H, W = g0.shape
        th, tw = int(tpl.shape[0] * scale), int(tpl.shape[1] * scale)
        t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
        best = None
        for ang in angles_deg:
            if ang == 0.0:
                g = g0
            else:
                M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
                g = cv2.warpAffine(g0, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if best is None or val > best[0]:
                best = (float(val), float(ang))
        return best[1], best[0]


def build_bank(frames_json, out_dir, files):
    """Cut templates from reference frames whose oracle boxes are known."""
    os.makedirs(out_dir, exist_ok=True)
    rows = {f["file"]: f for f in json.load(open(frames_json))["frames"]}
    spec = []
    for name in files:
        f = rows[name]
        rgb = cv2.cvtColor(cv2.imread(f["image"]), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
        crop = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        out = f"tpl_{os.path.splitext(name)[0]}.png"
        cv2.imwrite(os.path.join(out_dir, out), crop)
        spec.append({"file": out, "source": f["image"], "bbox": f["bbox"], "depth_m": f["depth_m"]})
    json.dump({"templates": spec}, open(os.path.join(out_dir, "templates.json"), "w"), indent=1)
    return spec
