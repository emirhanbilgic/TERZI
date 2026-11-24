#!/usr/bin/env python3
"""Self-generated training data for the LAYER fine-tune (finetune_masked.py), FT-pen style (datagen_main).

For every pilot sample with >= 1 visible single garment (<= 4 garments; single-layer outfits get the body term only), the FROZEN per-garment model
(train_masked.py checkpoints, 6 views) generates every single garment and the ALL setting
exactly as at inference (SS -> coords, SLat 12-step CFG with a stored noise). We keep:

  <out>/<s>/cond.npz     tok_rgb (L,C), tok_body (4,L',C) pooled x2, tok_mask_<g> (L,C)  fp16
  <out>/<s>/lat_<g>.npz  coords, feats (raw, un-normalised), eps                        g = g0.. / all
  <out>/<s>/body.npz     body_cube_v / body_cube_f  (watertight GT SMPL-X in the [-0.5,0.5]^3 cube)
  <out>/<s>/order.npz    per overlapping pair gi|gj: GT overlap points (metric px), smoothed share saying gi is OUTER
                         there, and the pair's overall share (LOCAL order: tucked / half-tucked tops)
  <out>/<s>/layers.json  settings, GT layer order of overlapping pairs, frozen-model stats (written LAST)

GT layer order (pairs): on the GT simulated garments (pergarment_split order), the vertices of
one garment within 3 px of the other vote which one is farther from the body there. A pair
needs >= 30 overlapping votes; the garments' body-distance ranking alone gets tucked vs untucked
tops wrong.

The frozen stats ARE the step-0 measurement: body penetration of each generated garment and,
per ordered pair, the fraction of overlapping vertices on the wrong side of the other layer.
"""
import argparse, io, json, os, sys, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent; _GBC = _HERE.parent; _WD = _GBC.parent
sys.path[:0] = [str(_HERE), str(_GBC), str(_GBC / "data"), str(_WD), str(_WD / "TRELLIS.2")]
torch.set_grad_enabled(False)
from trellis2 import models as _models
from trellis2.modules import sparse as sp
from common import bake_cond, build_pipeline, to_metric
from terzi import render as BR
import crop_space as cs
import pergarment_split as PS
from pen_loss import PenaltyBody, fast_bvh, surface_vertex_mask
from layer_loss import LayerSurface, layer_penalty, LocalOrder

# data roots (training/data/README.md); override with the environment variables
HM = Path(os.environ.get("TERZI_DATA", _WD / "two_branch_data"))                       # whole-outfit latents, crops, bodies
PG = Path(os.environ.get("TERZI_PERGARMENT", _WD / "pg/pergarment"))               # pergarment_prep.py output
PJ = Path(os.environ.get("TERZI_PROJECTS", _WD / "reproj_garment"))   # reprojected meshes
IDS_JSON = Path(os.environ.get("TERZI_IDS", _WD / "splits.json"))            # {"train": [...], "val": [...]}
SS_CFG = "configs/ss.json"; SLAT_CFG = "configs/slat.json"
MAX_GARMENTS = 4; MIN_VOTES = 30; VOTE_PX = 3.0


def load(cfg_path, path, eight):
    cfg = json.load(open(_WD / cfg_path))["models"]["denoiser"]
    if eight:
        cfg["args"]["in_channels"] = cfg["args"]["out_channels"] = 8
    m = getattr(_models, cfg["name"])(**cfg["args"]); sd = torch.load(_WD / path, map_location="cpu")
    miss, unexp = m.load_state_dict({x[9:]: v for x, v in sd.items() if x.startswith("denoiser.")}, strict=False)
    assert set(miss) <= {"rope_phases"} and not unexp, (miss, unexp)
    assert sd["view_embed"].shape[0] == 6, "needs a per-garment (6-view) checkpoint"
    return m.cuda().eval(), sd["view_embed"].cuda()


def pool(tok, f=2):
    N, L, C = tok.shape; g = int(round((L - 5) ** 0.5))
    p = F.avg_pool2d(tok[:, 5:].reshape(N, g, g, C).permute(0, 3, 1, 2), f).permute(0, 2, 3, 1).reshape(N, -1, C)
    return torch.cat([tok[:, :5], p], 1)


def cond6(tok_rgb, tok_body, tok_mask, ve):
    """= train_masked.BodyCondDenoiser.cond_tokens: RGB + mask full res, body pre-pooled."""
    e = ve.to(tok_rgb.dtype)
    return torch.cat([tok_rgb + e[0], (tok_body + e[1:5, None]).reshape(-1, tok_rgb.shape[-1]), tok_mask + e[5]], 0)[None]


def gt_local_order(garments, body_v, keep, k_smooth=32, max_pts=6000, seed=0):
    """Per overlapping GT pair (i < j, both in keep): overlap points (crop space) and the share of the k nearest
    overlap votes saying i is OUTER there -> {(i, j): (pts, f_i, maj_i, n)}. Same votes as gt_pairs."""
    bt = cKDTree(body_v); out = {}; rng = np.random.default_rng(seed)
    db = [bt.query(np.asarray(g["mesh"].vertices), k=1)[0] for g in garments]
    trees = [cKDTree(np.asarray(g["mesh"].vertices)) for g in garments]
    for i in range(len(garments)):
        for j in range(i + 1, len(garments)):
            if i not in keep or j not in keep:
                continue
            vi, vj = np.asarray(garments[i]["mesh"].vertices), np.asarray(garments[j]["mesh"].vertices)
            di, ni = trees[j].query(vi, k=1); dj, nj = trees[i].query(vj, k=1)
            mi, mj = di < VOTE_PX, dj < VOTE_PX
            if mi.sum() + mj.sum() < MIN_VOTES:
                continue
            pts = np.concatenate([0.5 * (vi[mi] + vj[ni[mi]]), 0.5 * (vj[mj] + vi[nj[mj]])])
            lab = np.concatenate([db[i][mi] > db[j][ni[mi]], db[i][nj[mj]] > db[j][mj]]).astype(np.float32)
            _, nb = cKDTree(pts).query(pts, k=min(k_smooth, len(pts)))
            f = lab[nb].mean(1) if nb.ndim == 2 else lab
            sel = rng.permutation(len(pts))[:max_pts]
            out[(i, j)] = (pts[sel], f[sel].astype(np.float32), float(lab.mean()), len(pts))
    return out


def gt_pairs(garments, body_v):
    """[(inner, outer, votes)] for overlapping GT garment pairs (crop space)."""
    bt = cKDTree(body_v); out = []
    db = [bt.query(np.asarray(g["mesh"].vertices), k=1)[0] for g in garments]
    trees = [cKDTree(np.asarray(g["mesh"].vertices)) for g in garments]
    for i in range(len(garments)):
        for j in range(i + 1, len(garments)):
            vi, vj = np.asarray(garments[i]["mesh"].vertices), np.asarray(garments[j]["mesh"].vertices)
            di, ni = trees[j].query(vi, k=1); dj, nj = trees[i].query(vj, k=1)
            mi, mj = di < VOTE_PX, dj < VOTE_PX
            j_out = int((db[j][ni[mi]] > db[i][mi]).sum() + (db[j][mj] > db[i][nj[mj]]).sum())
            n = int(mi.sum() + mj.sum())
            if n >= MIN_VOTES:
                out.append((i, j, n) if j_out * 2 >= n else (j, i, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss", required=True); ap.add_argument("--slat", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--n", type=int, default=0, help="max samples of the split (0 = all eligible)")
    ap.add_argument("--out", required=True); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--total-shards", type=int, default=1)
    ap.add_argument("--tokens-only", action="store_true", help="only the condition tokens (cond.npz), body.npz and layers.json {settings}: "
                    "no generation, no GT order -- the inputs distill/distill.py needs (targets come from the teacher on the fly)")
    ap.add_argument("--ids-file", default=None, help="only these outfit ids (one per line, in this order) instead of the whole split: "
                    "skips opening split.json for every id of the split, which is slow where NFS metadata is slow")
    a = ap.parse_args()
    out = Path(a.out) if os.path.isabs(a.out) else _WD / a.out
    ids = [l.strip() for l in open(a.ids_file) if l.strip()] if a.ids_file else json.load(open(IDS_JSON))[a.split]
    elig = []
    if a.ids_file:                                             # pre-filtered list: check eligibility lazily, per outfit, in this shard only
        ids = ids[:a.n] if a.n else ids
        elig = [(s, None) for s in ids]; ids = []
    for s in ids:
        try:
            g = json.load(open(PG / "masks" / s / "split.json"))["garments"]
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        sing = [f"g{k}" for k, x in enumerate(g) if x["single"]]
        if 1 <= len(sing) and len(g) <= MAX_GARMENTS:          # single-layer outfits too: body term only
            elig.append((s, sing))
    if a.n:
        elig = elig[:a.n]
    mine = elig[a.shard::a.total_shards]
    print(f"{len(elig)} eligible {a.split} samples, this shard {len(mine)}", flush=True)

    pipe = build_pipeline(); dec = pipe.models["shape_slat_decoder"]; dec.set_resolution(512)
    ssdec = pipe.models["sparse_structure_decoder"]; dino = pipe.image_cond_model
    MEAN = torch.tensor(pipe.shape_slat_normalization["mean"], device="cuda")[None]
    STD = torch.tensor(pipe.shape_slat_normalization["std"], device="cuda")[None]
    ssp, slp = dict(pipe.sparse_structure_sampler_params), dict(pipe.shape_slat_sampler_params)
    (ssm, ve_ss), (slm, ve_sl) = ((None, None), (None, None)) if a.tokens_only else (load(SS_CFG, a.ss, True), load(SLAT_CFG, a.slat, False))
    flip = None

    def st(c, f):
        return sp.SparseTensor(coords=torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), torch.as_tensor(c).int()], 1).cuda(),
                               feats=torch.as_tensor(f).float().cuda())

    for s, sing in mine:
        d = out / s
        if (d / "layers.json").exists():
            continue
        if sing is None:
            try:
                g = json.load(open(PG / "masks" / s / "split.json"))["garments"]
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            sing = [f"g{k}" for k, x in enumerate(g) if x["single"]]
            if not (1 <= len(sing) and len(g) <= MAX_GARMENTS):
                continue
        try:
            gv, gf, body, _ = PS.load_sample(PJ, s)
            garments = PS.split_outfit(gv, gf, np.asarray(body.vertices))
            g_all = json.load(open(PG / "masks" / s / "split.json"))["garments"]
            assert len(garments) == len(g_all), "split mismatch"
            pairs = [] if a.tokens_only else [p for p in gt_pairs(garments, np.asarray(body.vertices)) if f"g{p[0]}" in sing and f"g{p[1]}" in sing]
            LO = {} if a.tokens_only else gt_local_order(garments, np.asarray(body.vertices), {int(x[1:]) for x in sing})
            _, b = bake_cond(HM / "shape_garment_aligned/render_cond" / s / "000.png", 512, "jpeg", 95)
            rgb = torch.from_numpy(np.array(Image.open(io.BytesIO(b)).convert("RGB"))).permute(2, 0, 1).float()[None].cuda() / 255
            bd = np.load(HM / "shape_body_aligned/shape_latent" / f"{s}.npz"); bm = dec(st(bd["coords"], bd["feats"]))[0]
            settings = sing + (["all"] if len(sing) > 1 else [])
            masks = torch.stack([torch.from_numpy(np.array(Image.open(PG / "masks" / s / f"{g}.png")) > 127).float() for g in settings]).cuda()
            tok = dino(torch.cat([rgb, BR.render_views(bm.vertices, bm.faces), masks[:, None].expand(-1, 3, -1, -1)]))
            tok_rgb, tok_body, tok_m = tok[0], pool(tok[1:5]), tok[5:]
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "cond.npz", tok_rgb=tok_rgb.half().cpu().numpy(), tok_body=tok_body.half().cpu().numpy(),
                     **{f"tok_mask_{g}": tok_m[i].half().cpu().numpy() for i, g in enumerate(settings)})
            bcv = cs.to_cube(np.asarray(body.vertices, np.float64)).astype(np.float32)
            np.savez(d / "body.npz", body_cube_v=bcv, body_cube_f=np.asarray(body.faces, np.int32))
            if a.tokens_only:
                json.dump({"settings": settings, "tokens_only": True}, open(d / "layers.json", "w")); continue
            raw, flip = fast_bvh(to_metric(torch.from_numpy(bcv).cuda()), torch.from_numpy(np.asarray(body.faces, np.int64)).cuda(), flip)
            pb = PenaltyBody(raw)
            ordz = {}
            for (i, j), (pts, f_i, maj, n) in LO.items():
                ordz[f"g{i}|g{j}_pts"] = to_metric(torch.from_numpy(cs.to_cube(pts.astype(np.float64)).astype(np.float32)).cuda()).cpu().numpy()
                ordz[f"g{i}|g{j}_f"] = f_i; ordz[f"g{i}|g{j}_maj"] = np.float32(maj)
            np.savez(d / "order.npz", **ordz)
            meshes, stats = {}, {"body_pen": {}, "n_cells": {}}
            for i, g in enumerate(settings):
                seed = zlib.crc32(f"{s}/{g}".encode()) & 0x7FFFFFFF; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                c = cond6(tok_rgb, tok_body, tok_m[i], ve_ss)
                z = pipe.sparse_structure_sampler.sample(ssm, torch.randn(1, 8, 16, 16, 16, device="cuda"), cond=c,
                                                         neg_cond=torch.zeros_like(c), **ssp, verbose=False).samples
                coords = torch.argwhere(F.max_pool3d((ssdec(z) > 0).float(), 2, 2) > 0.5)[:, [0, 2, 3, 4]].int()
                if not len(coords):
                    raise ValueError(f"empty SS for {g}")
                c = cond6(tok_rgb, tok_body, tok_m[i], ve_sl)
                eps = torch.randn(coords.shape[0], slm.in_channels, device="cuda")
                zl = pipe.shape_slat_sampler.sample(slm, sp.SparseTensor(feats=eps, coords=coords), cond=c,
                                                    neg_cond=torch.zeros_like(c), **slp, verbose=False).samples
                feats = zl.feats.float() * STD + MEAN
                np.savez(d / f"lat_{g}.npz", coords=coords[:, 1:].cpu().numpy().astype(np.int16),
                         feats=feats.half().cpu().numpy(), eps=eps.half().cpu().numpy())
                m = dec(sp.SparseTensor(coords=coords, feats=feats))[0]
                vv = to_metric(m.vertices.float()); mk = surface_vertex_mask(m.faces.long(), vv.shape[0], "cuda")
                meshes[g] = (vv, m.faces.long(), mk)
                stats["body_pen"][g] = float((pb.bb.signed(vv[mk]) < 0).float().mean()); stats["n_cells"][g] = int(len(coords))
            stats["layer_cross"] = {}
            for (i, j), (_, _, maj, n) in LO.items():
                gi, gj = f"g{i}", f"g{j}"
                lo = LocalOrder(torch.from_numpy(ordz[f"{gi}|{gj}_pts"]).cuda(), torch.from_numpy(ordz[f"{gi}|{gj}_f"]).cuda(), maj)
                si = LayerSurface(meshes[gi][0], meshes[gi][1], pb); sj = LayerSurface(meshes[gj][0], meshes[gj][1], pb)
                qi, qj = meshes[gi][0][meshes[gi][2]], meshes[gj][0][meshes[gj][2]]
                _, st_i = layer_penalty(qi, sj, outer=lo.want(qi, True)); _, st_j = layer_penalty(qj, si, outer=lo.want(qj, False))
                stats["layer_cross"][f"{gi}|{gj}"] = {"i_cross": st_i["cross"], "i_overlap": st_i["overlap"],
                                                      "j_cross": st_j["cross"], "j_overlap": st_j["overlap"], "maj_i_outer": maj, "n_votes": n,
                                                      "minority": float(min(maj, 1 - maj))}
            json.dump({"settings": settings, "pairs": [[f"g{i}", f"g{j}", n] for i, j, n in pairs],
                       "local_pairs": [[f"g{i}", f"g{j}", v[3]] for (i, j), v in LO.items()], "frozen": stats,
                       "ss": a.ss, "slat": a.slat}, open(d / "layers.json.tmp", "w"), indent=1)
            os.replace(d / "layers.json.tmp", d / "layers.json")
            print(f"[ok] {s} {settings} pairs {[(i, j) for i, j, _ in pairs]} "
                  f"cross {[round(0.5 * (v['i_cross'] + v['j_cross']), 3) for v in stats['layer_cross'].values()]}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[fail] {s}: {e}", flush=True)


if __name__ == "__main__":
    main()
