#!/usr/bin/env python3
"""FT-pen training samples (finetune_main.py format) for a 5-view garment model (RGB + 4 body renders, NO mask, e.g. aligned_v1),
generated on the BEDLAM2.0 outfits prepared by datagen_masked.py.

Reuses, per outfit <s> of --layers (datagen_masked output):
  cond.npz :: tok_rgb, tok_body   the DINOv3 tokens of the RGB crop and the 4 body normal renders (decoded body latent),
                                  pooled 2x2 for the body, before any view_embed -- the exact input path of datagen_main
  body.npz :: body_cube_v/f       the WATERTIGHT GT SMPL-X in the crop cube (what the renders show; the penalty body)
and generates the model's OWN sample of the whole outfit (SS 12 steps CFG 7.5 -> coords -> SLat 12 steps CFG 7.5), with
datagen_main's seeds (crc32(<s>:ss), crc32(<s>:garment)). Writes exactly what finetune_main.py reads:
  latents_broken/<s>.npz  coords (N,3 uint8), feats (raw), eps (normalised-space SLat noise)
  cond_cache/<s>.npz      tok_rgb, tok_body
  body/<s>.npz            body_cube_v, body_cube_f
  rows/<s>.json           penetration read-out of the sample (resume marker)
  python training/finetune/datagen_main.py --ids <id file> --layers <datagen_masked.py --tokens-only output> --out <dir> --ss .. --slat ..
"""
import argparse, json, os, sys, time, zlib
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__)); _GBC = os.path.dirname(_HERE); _WD = os.path.dirname(_GBC)
sys.path[:0] = [_HERE, _GBC, _WD, os.path.join(_WD, "TRELLIS.2")]
import numpy as np, torch, torch.nn.functional as F
from trellis2 import models as _models
from trellis2.modules import sparse as sp
from common import build_pipeline, to_metric
from pen_loss import fast_bvh

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True, help="text file, one outfit id per line"); ap.add_argument("--layers", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--ss", required=True); ap.add_argument("--slat", required=True)
ap.add_argument("--ss-steps", type=int, default=12); ap.add_argument("--slat-steps", type=int, default=12)
ap.add_argument("--ss-cfg", type=float, default=7.5); ap.add_argument("--slat-cfg", type=float, default=7.5)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
a = ap.parse_args(); dev = "cuda"; os.chdir(_WD); out = Path(a.out); L = Path(a.layers)
for d in ("latents_broken", "cond_cache", "body", "rows"):
    (out / d).mkdir(parents=True, exist_ok=True)
ids = [l.strip() for l in open(a.ids) if l.strip()][a.shard::a.nshard]
todo = [s for s in ids if not (out / "rows" / f"{s}.json").exists()]
print(f"shard {a.shard}/{a.nshard}: {len(todo)}/{len(ids)} outfits -> {out}", flush=True)
if not todo:
    sys.exit(0)

pipe = build_pipeline()
dec = pipe.models["shape_slat_decoder"]; dec.set_resolution(512); dec.eval(); ssdec = pipe.models["sparse_structure_decoder"]
MEAN = torch.tensor(pipe.shape_slat_normalization["mean"], device=dev)[None]; STD = torch.tensor(pipe.shape_slat_normalization["std"], device=dev)[None]
slat_params = dict(pipe.shape_slat_sampler_params); ss_params = dict(pipe.sparse_structure_sampler_params)
ss_params.update(steps=a.ss_steps, guidance_strength=a.ss_cfg); slat_params.update(steps=a.slat_steps, guidance_strength=a.slat_cfg)
for k in [k for k in pipe.models if "flow" in k]:
    del pipe.models[k]
pipe.image_cond_model = None; torch.cuda.empty_cache()
PX_MM = 1940.0 / 512.0


def load(cfg_path, ckpt, ch=None):
    cfg = json.load(open(cfg_path))["models"]["denoiser"]
    if ch:
        cfg["args"]["in_channels"] = cfg["args"]["out_channels"] = ch
    m = getattr(_models, cfg["name"])(**cfg["args"]); sd = torch.load(ckpt, map_location="cpu")
    miss, unexp = m.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("denoiser.")}, strict=False)
    assert set(miss) <= {"rope_phases"} and not unexp, (miss, unexp)
    assert sd["view_embed"].shape[0] == 5, "needs a 5-view (no mask) checkpoint"
    return m.to(dev).eval(), sd["view_embed"].to(dev)


slat, VE_SLAT = load("configs/slat.json", a.slat)
ss, VE_SS = load("configs/ss.json", a.ss, ch=8)
print(f"SS {a.ss} {ss_params}\nSLat {a.slat} {slat_params}", flush=True)


def cond_tokens(tok_rgb, tok_body, ve):
    e = ve.to(tok_rgb.dtype); body = tok_body + e[1:, None, :]
    return torch.cat([tok_rgb + e[0], body.reshape(-1, body.shape[-1])], 0)[None]


def seed(s, tag):
    v = zlib.crc32(f"{s}:{tag}".encode()) & 0xFFFFFFFF; torch.manual_seed(v); torch.cuda.manual_seed_all(v)


ok = err = 0; t_start = time.time(); flip = None
for i, s in enumerate(todo):
    t0 = time.time()
    try:
        cz = np.load(L / s / "cond.npz"); bz = np.load(L / s / "body.npz")
        tok_rgb = torch.as_tensor(cz["tok_rgb"]).to(dev).float(); tok_body = torch.as_tensor(cz["tok_body"]).to(dev).float()
        with torch.no_grad():
            seed(s, "ss"); c_ss = cond_tokens(tok_rgb, tok_body, VE_SS)
            z = pipe.sparse_structure_sampler.sample(ss, torch.randn(1, 8, 16, 16, 16, device=dev), cond=c_ss,
                                                     neg_cond=torch.zeros_like(c_ss), **ss_params, verbose=False).samples
            coords = torch.argwhere(F.max_pool3d((ssdec(z) > 0).float(), 2, 2) > 0.5)[:, [0, 2, 3, 4]].int()
            if len(coords) == 0:
                raise RuntimeError("empty garment occupancy")
            seed(s, "garment"); c_sl = cond_tokens(tok_rgb, tok_body, VE_SLAT)
            eps = torch.randn(coords.shape[0], slat.in_channels, device=dev)
            zg = pipe.shape_slat_sampler.sample(slat, sp.SparseTensor(feats=eps, coords=coords), cond=c_sl,
                                                neg_cond=torch.zeros_like(c_sl), **slat_params, verbose=False).samples
            zg_raw = zg.replace(zg.feats * STD + MEAN); mg = dec(zg_raw)[0]
            bb, flip = fast_bvh(to_metric(torch.as_tensor(bz["body_cube_v"]).float().to(dev)), torch.as_tensor(bz["body_cube_f"].astype(np.int64)).to(dev), flip)
            sd_ = bb.signed(to_metric(mg.vertices.float()))
            pen_any = float((sd_ < 0).float().mean()); pen_deep = float((sd_ < -2.0 / PX_MM).float().mean())
        np.savez(out / "latents_broken" / f"{s}.npz", coords=zg.coords[:, 1:].cpu().numpy().astype(np.uint8),
                 feats=zg_raw.feats.float().cpu().numpy(), eps=eps.float().cpu().numpy())
        np.savez(out / "cond_cache" / f"{s}.npz", tok_rgb=cz["tok_rgb"], tok_body=cz["tok_body"])
        np.savez(out / "body" / f"{s}.npz", body_cube_v=bz["body_cube_v"].astype(np.float32), body_cube_f=bz["body_cube_f"].astype(np.int32),
                 body_source=np.array("gt"))
        json.dump({"sha": s, "cells": int(len(coords)), "pen_any": pen_any, "pen_deep": pen_deep, "sec": time.time() - t0},
                  open(out / "rows" / f"{s}.json", "w"))
        ok += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t_start
            print(f"[{i + 1}/{len(todo)}] ok {ok} err {err}  {el / (i + 1):.2f}s/outfit  last pen_any {100 * pen_any:.2f}%", flush=True)
    except Exception as e:
        err += 1; print(f"[err] {s}: {e!r}", flush=True)
print(f"DATAGEN_MAIN_DONE ok {ok} err {err}", flush=True)
