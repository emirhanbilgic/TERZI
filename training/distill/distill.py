#!/usr/bin/env python3
"""One-step students (paper Sec. 3.4, supplementary Sec. 6.2.5): reflow, then one-step distillation, of a garment SLat or SS DiT.
--kind ss: the 8-channel dense 16^3 structure latent; the student starts from pure noise (to-steps 1), so only the cached
condition tokens are used (no SS data latents needed).

Data (--data, --latents):
  <data>/<s>/cond.npz         DINOv3 tokens: tok_rgb (L,C), tok_body (4,L',C) pooled 2x2, tok_mask_<g> (datagen_masked.py --tokens-only)
  <latents>/<s>__<g>.npz      GT shape latent of garment g ('all' = the whole outfit): coords + feats (only the cells are used)
Two model families (the checkpoint's view_embed decides):
  5 views (aligned_v1 family): cond = RGB + 4 body views; x0 = the WHOLE outfit ('all', or the single garment)
  6 views (masked_best family): cond = RGB + 4 body views + garment mask; x0 = one garment or 'all' (--p-all)
Methods (CFG baked in when --teacher-cfg on; the student then samples with guidance 1.0 and --to-steps steps):
  segment  the student jumps one segment of the teacher's --from-steps guided Euler grid in ONE step, from the GT latent
           noised to the segment start (--to-steps 1: from pure noise to the teacher's sample = direct 1-step distillation)
  reflow   the teacher samples noise -> x0 with --from-steps guided steps; the student learns the STRAIGHT line between that
           noise and that sample at t ~ U(0, 1) (rectified flow on the teacher's own couplings; the GT latent only gives
           the cells). A reflowed model is then segment-distilled with --teacher-cfg off (CFG is already inside it).
Checkpoints: {denoiser.*, view_embed}, the format of the training scripts and of terzi.TerziPipeline.

Paper recipe, per stage (SLat: 6000 + 6000 updates; SS: --kind ss, 3000 + 3000), from the fine-tuned teacher:
  torchrun --nproc_per_node=1 training/distill/distill.py --teacher <teacher ckpt> --method reflow --teacher-cfg on \
      --from-steps 12 --steps 6000 --data <tokens> --out results/reflow
  torchrun --nproc_per_node=1 training/distill/distill.py --teacher results/reflow/ckpts/denoiser_last.pt --method segment \
      --teacher-cfg off --from-steps 12 --to-steps 1 --steps 6000 --max-cells 1500 --data <tokens> --out results/student
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)); _GBC = os.path.dirname(_HERE); _WD = os.path.dirname(_GBC)
sys.path[:0] = [_HERE, _GBC, _WD, os.path.join(_WD, "TRELLIS.2")]
import ddp_setup as D   # FIRST
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import torch.distributed as dist
from trellis2 import models as _models
from trellis2.modules import sparse as sp
from common import build_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["slat", "ss"], default="slat", help="ss: the 8-channel dense 16^3 structure latent (pure noise start, no data latent needed)")
ap.add_argument("--teacher", required=True); ap.add_argument("--init", default=None, help="student init (default: the teacher)")
ap.add_argument("--out", required=True)
ap.add_argument("--method", choices=["segment", "reflow"], required=True)
ap.add_argument("--from-steps", type=int, default=12); ap.add_argument("--to-steps", type=int, default=1)
ap.add_argument("--teacher-cfg", choices=["on", "off"], default="on")
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--grad-accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--save-every", type=int, default=250)
ap.add_argument("--data", required=True, help="condition tokens per outfit: training/finetune/datagen_masked.py --tokens-only output")
ap.add_argument("--latents", default="pg/pergarment/shape/shape_latent", help="garment shape latents <s>__<g>.npz (pergarment_prep.py + encoding)")
ap.add_argument("--p-all", type=float, default=0.2)
ap.add_argument("--max-cells", type=int, default=1500); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--student-depth", type=int, default=0, help="pruned student (not used in the paper): keep this many of the teacher's 30 blocks (15: 0,2,...,26,29; "
                "else evenly spaced incl. first and last); writes ckpts/arch.json. "
                "With --method segment --from-steps 1 --to-steps 1 --teacher-cfg off the target is the teacher's own 1-step velocity")
ap.add_argument("--warmup", type=int, default=0, help="linear learning-rate warm-up steps")
a = ap.parse_args()
assert a.from_steps % a.to_steps == 0
assert a.kind == "slat" or a.method == "reflow" or a.to_steps == 1, "ss segment distillation needs to-steps 1 (no SS data latents are used)"
dev = "cuda"; os.chdir(_WD); out = Path(a.out); (out / "ckpts").mkdir(parents=True, exist_ok=True)
C = json.load(open("configs/ss.json" if a.kind == "ss" else "configs/slat.json")); mcfg = C["models"]["denoiser"]
if a.kind == "ss":
    mcfg["args"]["in_channels"] = mcfg["args"]["out_channels"] = 8
pipe = build_pipeline()
sampler, spar = (pipe.sparse_structure_sampler, dict(pipe.sparse_structure_sampler_params)) if a.kind == "ss" else (pipe.shape_slat_sampler, dict(pipe.shape_slat_sampler_params))
guide = dict(guidance_strength=spar["guidance_strength"], guidance_rescale=spar.get("guidance_rescale", 0.0),
             guidance_interval=tuple(spar.get("guidance_interval", (0.0, 1.0))))
if a.teacher_cfg == "off":
    guide = dict(guide, guidance_strength=1.0)
rescale_t = float(spar.get("rescale_t", 1.0)); sigma_min = sampler.sigma_min
MEAN = torch.tensor(pipe.shape_slat_normalization["mean"], device=dev)[None]; STD = torch.tensor(pipe.shape_slat_normalization["std"], device=dev)[None]
for k in [k for k in pipe.models if "flow" in k]:
    del pipe.models[k]
pipe.image_cond_model = None; torch.cuda.empty_cache()


class Den(torch.nn.Module):
    def __init__(self, n_views, depth=None):
        super().__init__(); self.denoiser = getattr(_models, mcfg["name"])(**dict(mcfg["args"], **({"num_blocks": depth} if depth else {})))
        self.view_embed = torch.nn.Parameter(torch.zeros(n_views, mcfg["args"]["cond_channels"])); self.nv = n_views

    def cond(self, tok_rgb, tok_body, tok_mask=None):
        e = self.view_embed.to(tok_rgb.dtype); body = tok_body + e[1:5, None, :]
        parts = [tok_rgb + e[0], body.reshape(-1, body.shape[-1])] + ([tok_mask + e[5]] if self.nv == 6 else [])
        return torch.cat(parts, 0)[None]


def keep_blocks(n_keep, n=30):
    if n_keep == 15:
        return list(range(0, 28, 2)) + [n - 1]
    return [int(round(x)) for x in np.linspace(0, n - 1, n_keep)]


def prune_sd(sd, depth):
    """teacher state dict -> depth-block student: non-block weights copied, block i <- teacher block keep_blocks(depth)[i]."""
    blocks = keep_blocks(depth); new = {k: v for k, v in sd.items() if ".blocks." not in k}
    for i, j in enumerate(blocks):
        pre = f"denoiser.blocks.{j}."
        new.update({f"denoiser.blocks.{i}." + k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)})
    return new, blocks


def build(ckpt, depth=None, prune=False):
    sd = torch.load(ckpt, map_location="cpu")
    if prune:
        sd, kb = prune_sd(sd, depth); D.log(f"pruned init: blocks {kb}")
    m = Den(int(sd["view_embed"].shape[0]), depth)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert set(x.split(".")[-1] for x in miss) <= {"rope_phases"} and not unexp, (miss, unexp)
    return m.to(dev)


teacher = build(a.teacher).eval().requires_grad_(False)
last = out / "ckpts" / "denoiser_last.pt"; state_p = out / "ckpts" / "train_state.pt"; RESUME = last.exists() and state_p.exists()
SD = a.student_depth or None
student = build(str(last) if RESUME else (a.init or a.teacher), SD, prune=bool(SD) and not RESUME).train(); NV = student.nv
if D.IS_MAIN and SD:
    json.dump({"num_blocks": SD}, open(out / "ckpts" / "arch.json", "w"))
D.log(f"student params {sum(q.numel() for q in student.parameters()) / 1e6:.0f}M (teacher {sum(q.numel() for q in teacher.parameters()) / 1e6:.0f}M)")
if D.IS_MAIN and not RESUME:
    json.dump({"args": vars(a), "views": NV, "world": D.WORLD}, open(out / "run_config.json", "w"), indent=1)
D.log(f"{a.kind} {a.method} {a.from_steps}->{a.to_steps} | teacher {a.teacher} (cfg {a.teacher_cfg}, {NV} views) | init {a.init or a.teacher} | "
      f"accum {a.grad_accum} x {D.WORLD} | lr {a.lr} | steps {a.smoke or a.steps} | resume {RESUME}")

# ---------------------------------------------------------------- data
LAT = Path(a.latents)
IDS = sorted(p.name for p in Path(a.data).iterdir() if (p / "cond.npz").exists() and (p / "layers.json").exists())
D.log(f"outfits: {len(IDS)}")
rng = np.random.default_rng(1234 + D.RANK)


def draw():
    while True:
        s = IDS[rng.integers(len(IDS))]; d = Path(a.data) / s
        try:
            sett = json.load(open(d / "layers.json"))["settings"]; sing = [g for g in sett if g != "all"]
            if NV == 5:
                g = "all" if "all" in sett else sing[0]                 # v1: the whole outfit
            else:
                g = "all" if ("all" in sett and rng.uniform() < a.p_all) else sing[rng.integers(len(sing))]
            cz = np.load(d / "cond.npz")
            if a.kind == "ss":
                coords = None; x0 = torch.zeros(1, 8, 16, 16, 16, device=dev)       # never weighted: t = 1 start / teacher endpoint
            else:
                z = np.load(LAT / f"{s}__{g}.npz")
                if len(z["coords"]) > a.max_cells: continue
                coords = torch.cat([torch.zeros(len(z["coords"]), 1, dtype=torch.int32), torch.as_tensor(z["coords"].astype(np.int32))], 1).to(dev)
                x0 = (torch.as_tensor(z["feats"].astype(np.float32), device=dev) - MEAN) / STD
            tm = torch.as_tensor(cz[f"tok_mask_{g}"]).to(dev).float() if NV == 6 else None
            return coords, x0, torch.as_tensor(cz["tok_rgb"]).to(dev).float(), torch.as_tensor(cz["tok_body"]).to(dev).float(), tm
        except Exception:
            continue


def t_grid(n):
    t = np.linspace(1, 0, n + 1); return rescale_t * t / (1 + (rescale_t - 1) * t)


grid = t_grid(a.from_steps)
opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0); start = 1
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / a.warmup) if a.warmup else 1.0)
if RESUME:
    st = torch.load(state_p, map_location="cpu"); opt.load_state_dict(st["opt"]); start = int(st["step"]) + 1; D.log(f"resume at {start}")
total = a.smoke or a.steps; t0 = time.time(); ema = None; logf = open(out / "log.jsonl", "a") if D.IS_MAIN else None
for step in range(start, total + 1):
    opt.zero_grad(set_to_none=True); lacc = 0.0
    for _ in range(a.grad_accum):
        coords, x0, tr, tb, tm = draw()
        X = (lambda f: f) if a.kind == "ss" else (lambda f: sp.SparseTensor(coords=coords, feats=f))
        F_ = (lambda o: o) if a.kind == "ss" else (lambda o: o.feats)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cond_t = teacher.cond(tr, tb, tm); neg = torch.zeros_like(cond_t)
            eps = torch.randn_like(x0)
            if a.method == "segment":
                S = a.from_steps // a.to_steps; i = int(rng.integers(0, a.to_steps)) * S
                t_hi, t_lo = float(grid[i]), float(grid[i + S])
                xt = (1 - t_hi) * x0 + (sigma_min + (1 - sigma_min) * t_hi) * eps; x = xt
                for j in range(S):
                    tj, tj1 = float(grid[i + j]), float(grid[i + j + 1])
                    v = F_(sampler._inference_model(teacher.denoiser, X(x), tj, cond_t, neg_cond=neg, **guide)).float()
                    x = x - (tj - tj1) * v
                v_target = (xt - x) / max(t_hi - t_lo, 1e-8)
            else:
                x = (sigma_min + (1 - sigma_min) * 1.0) * eps
                for j in range(a.from_steps):
                    tj, tj1 = float(grid[j]), float(grid[j + 1])
                    v = F_(sampler._inference_model(teacher.denoiser, X(x), tj, cond_t, neg_cond=neg, **guide)).float()
                    x = x - (tj - tj1) * v
                x0t = x; t_hi = float(rng.uniform(0.0, 1.0))
                xt = (1 - t_hi) * x0t + (sigma_min + (1 - sigma_min) * t_hi) * eps
                v_target = (1 - sigma_min) * eps - x0t
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond_s = student.cond(tr, tb, tm)
            v_s = F_(student.denoiser(X(xt), torch.full((1,), 1000.0 * t_hi, device=dev), cond_s)).float()
        loss = ((v_s - v_target) ** 2).mean() / a.grad_accum; loss.backward(); lacc += float(loss)
    if D.DDP:
        for q in student.parameters():
            if q.grad is None: q.grad = torch.zeros_like(q)
            dist.all_reduce(q.grad); q.grad /= D.WORLD
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); sched.step()
    ema = lacc if ema is None else 0.98 * ema + 0.02 * lacc
    if D.IS_MAIN and (step % 20 == 0 or step <= 5):
        r = {"step": step, "loss": lacc, "ema": ema, "s_per_step": (time.time() - t0) / (step - start + 1)}
        logf.write(json.dumps(r) + "\n"); logf.flush(); print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
    if D.IS_MAIN and (step % a.save_every == 0 or step == total):
        sd = {k: v.detach().cpu() for k, v in student.state_dict().items()}
        torch.save(sd, out / "ckpts" / f"denoiser_step{step:07d}.pt"); torch.save(sd, str(last) + ".tmp"); os.replace(str(last) + ".tmp", last)
        torch.save({"step": step, "opt": opt.state_dict()}, str(state_p) + ".tmp"); os.replace(str(state_p) + ".tmp", state_p)
if D.IS_MAIN:
    (out / "DONE").write_text(f"{total}\n")
print("DISTILL_DONE", out, flush=True)
