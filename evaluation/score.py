#!/usr/bin/env python3
"""Score generated garments on 4D-DRESS or CloSe (paper Tabs. 2, 10, 13-17; supplementary Sec. 9).

    python evaluation/score.py --bench 4ddress --dataset <4D-DRESS root> --run <generate.py output> [--only <id>]
    python evaluation/score.py --bench close   --dataset <prep_close.py output> --run <generate.py output>

Per frame: the default post-processing (holes below 10 cm filled), one alignment of the complete prediction (our body + the
garment) to the complete scan, then the segmentation and GT label-transfer protocols (CD in mm, NC, F@1), penetration of the
garment into the conditioning body and the holes of the raw garment. Writes <run>/scores/<id>.json and <run>/summary.json.
"""
import argparse
import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M                                    # noqa: E402
from terzi import fill_holes                           # noqa: E402

BENCH = Path(__file__).resolve().parent / "benchmarks"
CLOTH_4DD = {3, 4, 5}                                  # 4D-DRESS labels: upper, lower, outer
NOT_GARMENT_CLOSE = [0, 1, 10, 12]                     # CloSe labels: hat, body, shoes, hair


def load_gt(bench, dataset, sample):
    """(scan vertices, faces, per-vertex labels with clothing in {3, 4, 5}, GT garment mesh)"""
    if bench == "4ddress":
        td = Path(dataset) / sample["take"]
        m = pickle.load(open(td / "Meshes_pkl" / f"mesh-f{sample['frame']}.pkl", "rb"))
        sv, sf = np.asarray(m["vertices"], np.float64), np.asarray(m["faces"], np.int64)
        sl = np.asarray(pickle.load(open(td / "Semantic" / "labels" / f"label-f{sample['frame']}.pkl", "rb"))["scan_labels"], np.int64)
        keep = np.isin(sl, list(CLOTH_4DD))
        fk = keep[sf].all(1)
        used = np.unique(sf[fk].ravel())
        remap = np.full(len(sv), -1, np.int64)
        remap[used] = np.arange(len(used))
        return sv, sf, sl, trimesh.Trimesh(sv[used], remap[sf[fk]], process=False)
    g = np.load(Path(dataset) / sample["id"] / "gt.npz")
    sv, sf, lab = g["V"].astype(np.float64), g["F"], g["L"].astype(int)
    sl = np.where(np.isin(lab, NOT_GARMENT_CLOSE), 1, 3)
    gt = trimesh.Trimesh(sv, sf, process=False).submesh([np.nonzero(np.isin(lab[sf], NOT_GARMENT_CLOSE, invert=True).sum(1) >= 2)[0]], append=True)
    return sv, sf, sl, gt


def score(bench, dataset, run, si, sample):
    sid = sample["id"]
    d = Path(run) / "glbs" / sid
    sv, sf, sl, gt = load_gt(bench, dataset, sample)
    load = lambda p: trimesh.load(str(p), force="mesh", process=False)
    body, raw = load(d / "cond_body.glb"), load(d / "pred_cloth.glb")
    garment = fill_holes(raw)
    full = trimesh.util.concatenate([body, garment])
    scan = trimesh.Trimesh(sv, sf, process=False)
    T = M.align_single(full, scan) if bench == "4ddress" else M.align_multi(full, scan)
    pen = M.penetration(garment, body, seed=zlib.crc32(sid.encode()) & 0x7FFFFFFF)
    rec = {"id": sid, "pen": pen, "holes": M.holes(raw)}
    lt = M.label_transfer_cloth(full.copy().apply_transform(T), sv, sf, sl, CLOTH_4DD)
    if lt is not None and len(lt.faces):
        rec["lt"] = M.sith_metrics(lt, gt, seed=si * 10 + 7)
    rec["seg"] = M.sith_metrics(garment.copy().apply_transform(T), gt, seed=si * 10 + 5)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["4ddress", "close"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    S = json.load(open(BENCH / ("4ddress_64.json" if a.bench == "4ddress" else "close_146.json")))
    S = [s if isinstance(s, dict) else {"id": s} for s in S]
    out = Path(a.run) / "scores"
    out.mkdir(parents=True, exist_ok=True)
    for si, s in enumerate(S):
        if (a.only and s["id"] not in a.only.split(",")) or not (Path(a.run) / "glbs" / s["id"] / "pred_cloth.glb").exists():
            continue
        f = out / f"{s['id']}.json"
        if not f.exists():
            json.dump(score(a.bench, a.dataset, a.run, si, s), open(f, "w"), default=float)
        r = json.load(open(f))
        print(s["id"], {k: round(r[k]["cd_mm"] / 10, 3) for k in ("seg", "lt") if k in r}, flush=True)
    R = [json.load(open(f)) for f in sorted(out.glob("*.json"))]
    mean = lambda p, k: float(np.mean([r[p][k] for r in R if p in r]))
    summ = {"n": len(R), **{f"{p}_{k}": (mean(p, "cd_mm") / 10 if k == "cd_cm" else mean(p, k)) for p in ("seg", "lt") for k in ("cd_cm", "nc", "fscore")},
            "pen_pct": 100 * float(np.mean([r["pen"] for r in R])), "holes": float(np.mean([r["holes"] for r in R]))}
    json.dump(summ, open(Path(a.run) / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
