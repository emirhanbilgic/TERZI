#!/usr/bin/env python3
"""Generate our garments on 4D-DRESS (64 frames) or CloSe (146 frontal renders), as for the paper's tables.

    python evaluation/generate.py --bench 4ddress --dataset <4D-DRESS root> --bodies <dir of <id>.npz> --model terzi --out runs/4dd_terzi
    python evaluation/generate.py --bench close   --dataset <prep_close.py output> --bodies <dir> --model terzi_masked --out runs/close_mk

Bodies: one <id>.npz per frame in the camera frame of the benchmark image ('vertices', optional 'faces' / 'pelvis' / 'joints';
SMPL-X faces and joints come from --smplx-model). The paper uses GVHMR on 4D-DRESS and PromptHMR on CloSe; no ground-truth
pose or shape (on 4D-DRESS, the crop window of every method is the box of the released SMPL-X fit). The masked models get the full-outfit mask: on 4D-DRESS rendered from the scan's garment labels (3, 4, 5) into
the crop, on CloSe the prep_close.py mask.
Output: <out>/glbs/<id>/{pred_cloth.glb, cond_body.glb} (raw garment, no post-processing: score.py applies it), cond.png.
4D-DRESS: meshes in the 4D-DRESS world frame, the body (and garment) Procrustes-fitted to the released SMPL-X fit, the garment
cut below the conditioning body's ankles; CloSe: camera frame, no cut. The scaled ICP of score.py makes the final alignment.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terzi import MODELS, TerziPipeline, load_body   # noqa: E402
from terzi import crop as C                          # noqa: E402

BENCH = Path(__file__).resolve().parent / "benchmarks"


def gt_cloth_mask_4dd(td, frame, K, R, T, box, labels=(3, 4, 5), S=512):
    """(S, S) bool: crop pixels whose front-most scan surface is clothing."""
    import nvdiffrast.torch as dr
    m = pickle.load(open(td / "Meshes_pkl" / f"mesh-f{frame}.pkl", "rb"))
    sv, sf = np.asarray(m["vertices"], np.float64), np.asarray(m["faces"], np.int64)
    sl = np.asarray(pickle.load(open(td / "Semantic" / "labels" / f"label-f{frame}.pkl", "rb"))["scan_labels"])
    Pc = (R @ sv.T).T + T
    u = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]
    k = S / box["size"]
    uc, vc = (u - (box["cx"] - box["size"] / 2)) * k, (v - (box["cy"] - box["size"] / 2)) * k
    z = Pc[:, 2]
    zn = (z - z.min()) / max(z.max() - z.min(), 1e-6) * 1.8 - 0.9
    pos = torch.tensor(np.stack([2 * uc / S - 1, 2 * vc / S - 1, zn, np.ones_like(zn)], 1), dtype=torch.float32, device="cuda")[None]
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), pos.contiguous(), torch.tensor(sf, dtype=torch.int32, device="cuda"), resolution=[S, S])
    own = torch.cat([torch.zeros(1, dtype=torch.bool), torch.from_numpy(np.isin(sl, list(labels))[sf].all(1))]).cuda()
    return own[rast[0, ..., 3].long()].cpu().numpy()


def smplx_joints(vertices, smplx_model):
    from terzi.body import _smplx_model
    return _smplx_model(smplx_model)["J_regressor"] @ np.asarray(vertices, np.float64)


def procrustes(src, dst, n=10_000, n_iter=5, seed=0):
    """similarity aligning mesh src to mesh dst (nearest-neighbour Umeyama)."""
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import umeyama
    p = trimesh.sample.sample_surface(src, n, seed=seed)[0]
    g = trimesh.sample.sample_surface(dst, n, seed=seed + 1)[0]
    tree, cur = cKDTree(g), p.copy()
    for _ in range(n_iter):
        s, Rm, t = umeyama(p, g[tree.query(cur)[1]])
        cur = (s * (Rm @ p.T)).T + t
    return s, Rm, t


def cut_below(mesh, y, margin=0.01):
    v = np.asarray(mesh.vertices)
    keep = v[:, 1] >= y - margin
    if keep.all() or not keep.any():
        return mesh
    fk = keep[mesh.faces].all(1)
    used = np.unique(mesh.faces[fk].ravel())
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(v[used], remap[mesh.faces[fk]], process=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["4ddress", "close"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--model", default="terzi", choices=list(MODELS))
    ap.add_argument("--ss", default=None)
    ap.add_argument("--slat", default=None)
    ap.add_argument("--smplx-model", default="data/SMPLX_NEUTRAL.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    pipe = TerziPipeline.from_pretrained(a.model, ss_path=a.ss, slat_path=a.slat)
    S = json.load(open(BENCH / ("4ddress_64.json" if a.bench == "4ddress" else "close_146.json")))
    S = [s if isinstance(s, dict) else {"id": s} for s in S]
    for s in S:
        sid = s["id"]
        if a.only and sid not in a.only.split(","):
            continue
        d = Path(a.out) / "glbs" / sid
        if (d / "pred_cloth.glb").exists():
            continue
        body = load_body(Path(a.bodies) / f"{sid}.npz", smplx_model=a.smplx_model)
        if a.bench == "4ddress":
            td = Path(a.dataset) / s["take"]
            cams = pickle.load(open(td / "Capture" / "cameras.pkl", "rb"))[s["view"]]
            K = np.asarray(cams["intrinsics"], np.float64)
            Rt = np.asarray(cams["extrinsics"], np.float64)
            R, T = Rt[:, :3], Rt[:, 3]
            image = np.asarray(Image.open(td / "Capture" / s["view"] / "images" / f"capture-f{s['frame']}.png").convert("RGB"))
            pmask = np.asarray(Image.open(td / "Capture" / s["view"] / "masks" / f"mask-f{s['frame']}.png").convert("L"))
            fit = trimesh.load(str(td / "SMPLX" / f"mesh-f{s['frame']}_smplx.ply"), force="mesh", process=False)
            box = C.crop_box((R @ np.asarray(fit.vertices).T).T + T, K)   # the crop all methods got: the released SMPL-X fit's box
            gmask = mcrop = None
            if pipe.masked:                       # full-outfit mask from the scan labels, rendered into the crop
                mcrop = gt_cloth_mask_4dd(td, s["frame"], K, R, T, box)
        else:
            w = Path(a.dataset) / sid
            K = np.asarray(json.load(open(w / "cam.json"))["K"], np.float64)
            image = np.asarray(Image.open(w / "input.jpg").convert("RGB"))
            pmask = np.asarray(Image.open(w / "mask_p1.png").convert("L"))
            gmask = np.asarray(Image.open(w / "gmask_p1_all.png").convert("L")) if pipe.masked else None
            mcrop = box = None
        out = pipe(image, pmask, K, body, garment_mask=gmask, mask_crop=mcrop, box=box, seed=sid, holes="none")
        garment, body_m = out.garment, out.body
        if a.bench == "4ddress":                  # world frame, Procrustes to the released SMPL-X fit, cut below the ankles
            to_w = lambda m: trimesh.Trimesh((R.T @ (np.asarray(m.vertices) - T).T).T, m.faces, process=False)
            garment, body_m = to_w(garment), to_w(body_m)
            sc, Rm, t = procrustes(body_m, fit)
            sim = lambda m: trimesh.Trimesh((sc * (Rm @ np.asarray(m.vertices).T)).T + t, m.faces, process=False)
            garment, body_m = sim(garment), sim(body_m)
            if len(body_m.vertices) == 10475:
                J = smplx_joints(body_m.vertices, a.smplx_model)
                garment = cut_below(garment, float(min(J[7, 1], J[8, 1])))
        d.mkdir(parents=True, exist_ok=True)
        garment.export(d / "pred_cloth.glb")
        body_m.export(d / "cond_body.glb")
        out.cond.save(d / "cond.png")
        print(sid, len(garment.vertices), flush=True)


if __name__ == "__main__":
    main()
