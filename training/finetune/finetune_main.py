#!/usr/bin/env python3
"""V3: fine-tune the garment SLat DiT with a DIFFERENTIABLE penetration penalty.

No learned head, no clamped targets, no extra inference cost.

Objective per step
------------------
  x_t   = (1-t) z_gen + (sigma + (1-sigma) t) eps          t ~ U(0.02, 0.98)
  v     = DiT(x_t, 1000 t, cond)
  L_fm  = || v - ((1-sigma) eps - z_gen) ||^2              ANCHOR: reflow on the
          frozen generator's OWN samples with their OWN noise, so the objective
          has the frozen model as an exact fixed point.  Nothing here asks the
          model to change; it only forbids drifting away from itself.
  x0hat = x_t - t v                                         one-step clean estimate
  L_pen = hinge( clearance - signed_distance( decode(x0hat), body ) )^2
  L     = L_fm + w_pen * L_pen

Why the anchor is the model's own output rather than the GT garment: the GT
BEDLAM garment was draped on the GT body, and the conditioning body here is a
GVHMR estimate that is ~34 mm away from it, so the GT target *actively teaches
penetration*.  Anchoring on the frozen generator makes "stay yourself" and
"get out of the body" the only two forces, and w_pen is the whole trade-off.

The penalty is measured against the WATERTIGHT placed SMPL-X mesh
(body/<sha>.npz :: body_cube_v/body_cube_f), never the decoded SC-VAE body
(~1300 components; distances to it are noisy).

Differentiability: see pen_loss.py.  mesh.vertices out of the SLat decoder is an
affine map of sigmoid(h[:, 0:3]); only the face WIRING is discrete.

Paper setting (TERZI, supplementary Tab. 6; the released checkpoint is step 100):
  torchrun --nproc_per_node=1 training/finetune/finetune_main.py --pairs <datagen_main.py output>/train \
      --val-pairs <...>/val --init <main SLat ckpt> --out results/ft_main --keep-steps
"""
import sys, os, json, time, argparse, math
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
_GBC = os.path.dirname(_HERE); _WD = os.path.dirname(_GBC)
sys.path[:0] = [_HERE, _GBC, _WD, os.path.join(_WD, "TRELLIS.2")]
import ddp_setup as D   # FIRST
import numpy as np, torch
import torch.distributed as dist
from trellis2 import models as _models
from trellis2.modules import sparse as sp
from rollout import rollout_x0                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True, help="datagen_main.py output (comma-separated for several roots)")
ap.add_argument("--val-pairs", default=None, help="default: last --val-n shas of --pairs")
ap.add_argument("--init", required=True, help="SLat checkpoint of the main model (train_main.py)")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--w-pen", type=float, default=30.0)
ap.add_argument("--anchor", choices=["self", "reflow"], default="self",
                help="self  : L = ||v - v_frozen(x_t)||^2 against a frozen copy of the init model. "
                     "The frozen model is then an EXACT stationary point of the anchor, so every "
                     "change is attributable to the penalty (w_pen=0 must be a no-op -- there is a "
                     "control run for that). "
                     "reflow: L = ||v - ((1-s)eps - z_gen)||^2, i.e. reflow on the generator's own "
                     "samples with their own noise. Also has the generator as a fixed point in "
                     "distribution, but NOT at init (the 12-step ODE is curved), so it moves the "
                     "model on its own.")
ap.add_argument("--w-anchor", type=float, default=1.0)
ap.add_argument("--p-uncond", type=float, default=0.1,
                help="fraction of steps done with a ZERO cond, anchor-only: sampling uses CFG 7.5, so "
                     "the unconditional branch must be pinned too or guidance drifts.")
ap.add_argument("--w-drift", type=float, default=0.2, help="val score = pen_tol + w_drift * drift_px")
# ---- ablation knobs
ap.add_argument("--ram-cap", type=int, default=24000,
                help="max samples held in the host-RAM cache (~4.3 MB each: the two DINO token "
                     "sets dominate). 20k samples = ~86 GB, which fits the 220 GB the job asks for; "
                     "a smaller cap makes every step re-read npz from Lustre and roughly triples "
                     "the step time.")
ap.add_argument("--max-train", type=int, default=0,
                help="subsample the training set to N pairs (0 = all). Tests whether the ~300-step "
                     "saturation is convergence or a data ceiling.")
ap.add_argument("--cfg-pen", type=float, default=0.0,
                help="apply the penalty to the CFG-COMBINED velocity v_u + s*(v_c - v_u) instead of the "
                     "conditional one (costs one extra no-grad-free forward). Sampling uses CFG 7.5, so "
                     "the conditional-only penalty is trained off-distribution. NOTE the sampler's "
                     "guidance_interval is [0.6,1.0], i.e. HIGH noise only, so this can only matter "
                     "together with a large --pen-tmax.")
ap.add_argument("--limb-w", type=float, default=0.0,
                help="extra weight on garment vertices whose nearest body point is on an arm or a leg "
                     "(0 = uniform). The diagnosis puts ~55%% of penetration on the 21%% of vertices in "
                     "the arms. weight = 1 + limb_w on limbs, 1 elsewhere.")
ap.add_argument("--clearance", type=float, default=1.5, help="crop px (1 px ~ 3.79 mm); 1.5 px ~ 5.7 mm (paper). "
                "The clearance is the main lever (supplementary Tab. 8); 2.0 px tears the surface of the main model.")
ap.add_argument("--deep-clip", type=float, default=2.0,
                help="px; deeper violations become linear. 2.0 px == the decoder's vertex-travel ceiling in x/y "
                     "(the dual vertex lives in sigmoid range [-0.5,1.5] -> 2/512 of the cube -> 2/512*INV_SCALE_UV "
                     "= 2.0 crop px ~ 7.6 mm; 3.33 px in z). A violation deeper than this CANNOT be fixed by moving "
                     "vertices at all, so letting it own the gradient only vents through surface deletion "
                     "(the failure that killed the 2026-06..08 collision-loss program).")
ap.add_argument("--pen-tmax", type=float, default=0.6,
                help="only apply the penalty for t <= this (x0hat is blurry at large t)")
ap.add_argument("--pen-tweight", action="store_true", help="scale the penalty by (1-t)")
ap.add_argument("--pen-rollout", type=int, default=1,
                help="Euler steps from t to 0 for the penalised clean estimate (1 = x_t - t v; CF/rollout.py)")
ap.add_argument("--pen-rollout-grad", choices=["all", "last"], default="all")
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--val-n", type=int, default=48)
ap.add_argument("--val-steps", type=int, default=12)
ap.add_argument("--patience", type=int, default=99)
ap.add_argument("--keep-steps", action="store_true")
ap.add_argument("--max-cells", type=int, default=1200, help="skip pathologically large samples")
a = ap.parse_args()
dev = "cuda"; SIGMA = 1e-5
os.chdir(_WD)
out = Path(a.out)
if D.IS_MAIN:
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(a), "world": D.WORLD}, open(out / "run_config.json", "w"), indent=1)
if D.DDP:
    dist.barrier()

C = json.load(open("configs/slat.json"))
mcfg = C["models"]["denoiser"]; norm = C["dataset"]["args"]["normalization"]
MEAN = torch.tensor(norm["mean"], dtype=torch.float32, device=dev)[None]
STD = torch.tensor(norm["std"], dtype=torch.float32, device=dev)[None]

torch.set_grad_enabled(True)
from common import BodyBVH, _tm, to_metric, mesh_openness              # noqa: E402
from pen_loss import PenaltyBody, penetration_penalty, surface_vertex_mask, fast_bvh  # noqa: E402


class BodyCondDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = getattr(_models, mcfg["name"])(**dict(mcfg["args"]))
        self.view_embed = torch.nn.Parameter(torch.zeros(5, mcfg["args"]["cond_channels"]))

    def cond(self, tok_rgb, tok_body):
        e = self.view_embed.to(tok_rgb.dtype)
        body = tok_body + e[1:, None, :]
        return torch.cat([tok_rgb + e[0], body.reshape(-1, body.shape[-1])], 0)[None]


model = BodyCondDenoiser()
sd = torch.load(a.init, map_location="cpu")
miss, unexp = model.load_state_dict(sd, strict=False)
assert set(m.split(".")[-1] for m in miss) <= {"rope_phases"} and not unexp, (miss, unexp)
model = model.to(dev).train()
params = list(model.parameters()); names = [n for n, _ in model.named_parameters()]
master = [p.detach().float().clone().requires_grad_(True) for p in params]
opt = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.0)

model0 = None
if a.anchor == "self":
    import copy
    model0 = copy.deepcopy(model).eval()
    for q in model0.parameters():
        q.requires_grad_(False)

dec = _models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(dev).eval()
dec.set_resolution(512)
# decoder params keep requires_grad=True: flex_gemm's submanifold-conv backward
# dereferences grad_weight unconditionally. They are not in the optimiser and
# their grads are dropped after every step.
D.log(f"init {a.init} | world {D.WORLD} x accum {a.grad_accum} | lr {a.lr} | w_pen {a.w_pen} "
      f"| clearance {a.clearance} px | params {sum(p.numel() for p in params)/1e9:.3f}B")

from trellis2.pipelines import samplers                                 # noqa: E402
from huggingface_hub import hf_hub_download                             # noqa: E402
pj = json.load(open(hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")))["args"]
SAMPLER = getattr(samplers, pj["shape_slat_sampler"]["name"])(**pj["shape_slat_sampler"]["args"])
SPARAMS = dict(pj["shape_slat_sampler"]["params"]); SPARAMS["steps"] = a.val_steps

# ------------------------------------------------------------------ data
# --pairs may name SEVERAL roots (comma-separated). The val set is always the tail
# of the FIRST root, so a data-scale comparison holds the val shas fixed.
ROOTS = [Path(x) for x in a.pairs.split(",") if x]
root = ROOTS[0]
WHERE = {}          # sha -> its root
ALL = []
for r in ROOTS:
    for p in sorted((r / "latents_broken").glob("*.npz")):
        if p.stem in WHERE:
            continue
        if (r / "cond_cache" / p.name).exists() and (r / "body" / p.name).exists():
            WHERE[p.stem] = r; ALL.append(p.stem)
FIRST = sorted(s for s in ALL if WHERE[s] == root)
VAL = FIRST[-a.val_n:] if a.val_pairs is None else sorted(
    p.stem for p in (Path(a.val_pairs) / "latents_broken").glob("*.npz"))[:a.val_n]
VROOT = root if a.val_pairs is None else Path(a.val_pairs)
for s_ in VAL:
    WHERE.setdefault(s_, VROOT)
TRAIN = [s for s in ALL if s not in set(VAL)]
if a.max_train and a.max_train < len(TRAIN):
    TRAIN = list(np.random.default_rng(0).permutation(TRAIN))[:a.max_train]
D.log(f"pairs {len(ALL)} over {len(ROOTS)} root(s) -> train {len(TRAIN)} / val {len(VAL)}  ({[str(r) for r in ROOTS]})")
assert len(TRAIN) >= 8, f"only {len(TRAIN)} training pairs"
RAM = {}
RAM_CAP = a.ram_cap


def item(sha, r=None):
    r = r or WHERE.get(sha, root)
    k = (str(r), sha)
    if k in RAM:
        return RAM[k]
    zb = np.load(r / "latents_broken" / f"{sha}.npz")
    cz = np.load(r / "cond_cache" / f"{sha}.npz")
    bz = np.load(r / "body" / f"{sha}.npz")
    e = {"coords": torch.as_tensor(zb["coords"].astype(np.int32)),
         "z": torch.as_tensor((zb["feats"] - MEAN.cpu().numpy()) / STD.cpu().numpy()).half(),
         "eps": torch.as_tensor(zb["eps"]).half(),
         "tok_rgb": torch.as_tensor(cz["tok_rgb"]), "tok_body": torch.as_tensor(cz["tok_body"]),
         "bv": torch.as_tensor(bz["body_cube_v"]).float(),
         "bf": torch.as_tensor(bz["body_cube_f"].astype(np.int64))}
    if len(RAM) >= RAM_CAP:
        RAM.pop(next(iter(RAM)))
    RAM[k] = e
    return e


BVH = {}
BVH_CAP = 512           # a cuBVH is ~2 MB of GPU; 20k of them is not an option
FLIP = [None]           # the SMPL-X winding test, done once and reused
LIMB_V = None
if a.limb_w > 0:
    # dominant skinning joint per SMPL-X vertex -> arm/leg mask. Joint ids:
    # 16/17 shoulder, 18/19 elbow, 20/21 wrist, 25.. hands ; 1/2 hip, 4/5 knee,
    # 7/8 ankle, 10/11 foot.
    from common import smplx_layer                                     # noqa: E402
    _lay = smplx_layer(dev)
    _dom = _lay.lbs_weights.argmax(dim=1).cpu().numpy()
    _limb = set([16, 17, 18, 19, 20, 21] + list(range(25, 55)) + [1, 2, 4, 5, 7, 8, 10, 11])
    LIMB_V = torch.tensor(np.isin(_dom, list(_limb)), dtype=torch.float32, device=dev)
    del _lay
    D.log(f"limb weighting on: {float(LIMB_V.mean()):.3f} of body vertices are limbs, "
          f"weight 1+{a.limb_w}")


def bvh_of(sha, e):
    if sha not in BVH:
        raw, FLIP[0] = fast_bvh(to_metric(e["bv"].to(dev)), e["bf"].to(dev), FLIP[0])
        b = PenaltyBody(raw)
        if LIMB_V is not None:
            b.limb_v = LIMB_V           # body_cube_v is the SMPL-X vertex order
        if len(BVH) >= BVH_CAP:
            BVH.pop(next(iter(BVH)))
        BVH[sha] = b
    return BVH[sha]


def pack(e):
    c = e["coords"].int()
    return torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), c], 1).to(dev)


# ------------------------------------------------------------------ validation
MYVAL = VAL[D.RANK::D.WORLD]
REF = {}


def _mesh_stats(m, body):
    vv = to_metric(m.vertices.float())
    mk = surface_vertex_mask(m.faces.long(), vv.shape[0], dev)
    v = vv[mk]
    s = body.bb.signed(v)
    f = m.faces.long()
    v0, v1, v2 = vv[f[:, 0]], vv[f[:, 1]], vv[f[:, 2]]
    area = float(0.5 * torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1).sum())
    return {"pen": float((s < 0).float().mean()),
            "pen_tol": float((s < -0.05 / 3.79).float().mean()),      # 0.05 mm
            "deep": float((s < -1.9 / 3.79).float().mean()),          # 1.9 mm
            "contact": float(((s >= -0.5) & (s < 0.5)).float().mean()),
            "so_med": float(s.median()), "area": area,
            "bnd": float(mesh_openness(vv.detach().cpu().numpy(), f.cpu().numpy()))}, v


def _chamfer(a_, b_, n=8000, chunk=256):
    """Symmetric point-to-SURFACE Chamfer in crop px between the fine-tuned and the
    frozen generator's decoded surface: the DRIFT guard. Both halves are kept -- the
    reference->current half is the only term that can see DELETED cloth, the failure
    mode every earlier collision fine-tune vented through.

    One side is subsampled, the other is used in FULL, so two samplings of the SAME
    surface score ~0 (subsampling both leaves a large density-dependent floor)."""
    g = torch.Generator(device=dev).manual_seed(0)

    def half(X, Y):
        X = X[torch.randperm(len(X), device=dev, generator=g)[:n]]
        o = 0.0
        for i in range(0, len(X), chunk):
            o += float(torch.cdist(X[i:i + chunk], Y).min(1).values.sum())
        return o / max(len(X), 1)
    return 0.5 * (half(a_, b_) + half(b_, a_))


@torch.no_grad()
def run_val():
    model.eval()
    acc = {k: 0.0 for k in ("pen", "pen_tol", "deep", "contact", "so_med", "bnd", "drift", "area_ratio")}
    for sha in MYVAL:
        e = item(sha, VROOT); coords = pack(e); body = bvh_of(sha, e)
        cond = model.cond(e["tok_rgb"].to(dev), e["tok_body"].to(dev))
        if sha not in REF:
            mr = dec(sp.SparseTensor(coords=coords, feats=e["z"].to(dev).float() * STD + MEAN))[0]
            st, vr = _mesh_stats(mr, body)
            REF[sha] = (st, vr.clone())
        noise = sp.SparseTensor(feats=e["eps"].to(dev).float(), coords=coords)
        z = SAMPLER.sample(model.denoiser, noise, cond=cond, neg_cond=torch.zeros_like(cond),
                           **SPARAMS, verbose=False).samples.feats
        m = dec(sp.SparseTensor(coords=coords, feats=z.float() * STD + MEAN))[0]
        st, v = _mesh_stats(m, body)
        for k in ("pen", "pen_tol", "deep", "contact", "so_med", "bnd"):
            acc[k] += st[k]
        acc["drift"] += _chamfer(v, REF[sha][1])
        acc["area_ratio"] += st["area"] / max(REF[sha][0]["area"], 1e-9)
    model.train()
    t = torch.tensor([acc[k] for k in acc], device=dev, dtype=torch.float64)
    if D.DDP:
        dist.all_reduce(t)
    return {k: float(t[i]) / len(VAL) for i, k in enumerate(acc)}


@torch.no_grad()
def ref_val():
    """Frozen-generator reference numbers on the same val set (for the report)."""
    model.eval()
    acc = {k: 0.0 for k in ("pen", "pen_tol", "deep", "contact", "so_med", "bnd")}
    for sha in MYVAL:
        e = item(sha, VROOT); coords = pack(e); body = bvh_of(sha, e)
        mr = dec(sp.SparseTensor(coords=coords, feats=e["z"].to(dev).float() * STD + MEAN))[0]
        st, vr = _mesh_stats(mr, body)
        REF[sha] = (st, vr.clone())
        for k in acc:
            acc[k] += st[k]
    model.train()
    t = torch.tensor([acc[k] for k in acc], device=dev, dtype=torch.float64)
    if D.DDP:
        dist.all_reduce(t)
    return {k: float(t[i]) / len(VAL) for i, k in enumerate(acc)}


def save(name):
    if D.IS_MAIN:
        p = out / "ckpts" / name
        torch.save({n: m.detach().to(q.dtype).cpu() for n, m, q in zip(names, master, params)}, str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)


# ------------------------------------------------------------------ train
logf = open(out / "log.jsonl", "a") if D.IS_MAIN else None
r0 = ref_val()
D.log("[ref] frozen generator on the val set: " + "  ".join(f"{k} {v:.4f}" for k, v in r0.items()))
v0 = run_val()
D.log("[val] step 0: " + "  ".join(f"{k} {v:.4f}" for k, v in v0.items()))
if D.IS_MAIN:
    logf.write(json.dumps({"step": 0, "ref": r0, "val": v0}) + "\n"); logf.flush()

rng = np.random.default_rng(1000 + D.RANK)
order, best, stalled, ema, t0 = [], float("inf"), 0, None, time.time()
for step in range(1, a.steps + 1):
    for p in params:
        p.grad = None
    tot = tot_fm = tot_pen = 0.0; st_last = {}
    for _ in range(a.grad_accum):
        if not order:
            order = list(rng.permutation(len(TRAIN)))
        sha = TRAIN[order.pop()]
        e = item(sha)
        if len(e["coords"]) > a.max_cells:
            continue
        coords = pack(e)
        z = e["z"].to(dev).float(); eps = e["eps"].to(dev).float()
        cond = model.cond(e["tok_rgb"].to(dev), e["tok_body"].to(dev))
        uncond_step = bool(rng.uniform() < a.p_uncond)
        if uncond_step:
            cond = torch.zeros_like(cond)
        t = float(rng.uniform(0.02, 0.98))
        x_t = (1 - t) * z + (SIGMA + (1 - SIGMA) * t) * eps
        tt = torch.full((1,), 1000.0 * t, device=dev)
        sten = sp.SparseTensor(coords=coords, feats=x_t)
        if a.anchor == "self":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                target = model0.denoiser(sten, tt, cond).feats.float()
        else:
            target = (1 - SIGMA) * eps - z
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model.denoiser(sp.SparseTensor(coords=coords, feats=x_t), tt, cond).feats
        v = v.float()
        l_fm = ((v - target) ** 2).mean()
        loss = a.w_anchor * l_fm
        l_pen = torch.zeros((), device=dev)
        if a.w_pen > 0 and t <= a.pen_tmax and not uncond_step:
            v_eff = v
            if a.cfg_pen > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v_u = model.denoiser(sp.SparseTensor(coords=coords, feats=x_t), tt,
                                         torch.zeros_like(cond)).feats
                v_eff = v_u.float() + a.cfg_pen * (v - v_u.float())
            if a.pen_rollout > 1:
                def _vel(f_, tk_):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return model.denoiser(sp.SparseTensor(coords=coords, feats=f_), torch.full((1,), 1000.0 * tk_, device=dev), cond).feats.float()
                zhat = rollout_x0(_vel, x_t, t, v_eff, a.pen_rollout, grad=a.pen_rollout_grad)
            else:
                zhat = x_t - t * v_eff
            mesh = dec(sp.SparseTensor(coords=coords, feats=zhat * STD + MEAN))[0]
            vv = to_metric(mesh.vertices.float())
            mk = surface_vertex_mask(mesh.faces.long(), vv.shape[0], dev)
            body_ = bvh_of(sha, e)
            rw = body_.limb_weight(vv[mk], a.limb_w) if a.limb_w > 0 else None
            l_pen, st_last = penetration_penalty(vv[mk], body_, clearance=a.clearance,
                                                 deep_clip=a.deep_clip, region_w=rw)
            w = a.w_pen * ((1.0 - t) if a.pen_tweight else 1.0)
            loss = loss + w * l_pen
        (loss / a.grad_accum).backward()
        tot += float(loss) / a.grad_accum
        tot_fm += float(l_fm) / a.grad_accum
        tot_pen += float(l_pen) / a.grad_accum
        dec.zero_grad(set_to_none=True)
    if D.DDP:
        buckets = {}
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            buckets.setdefault(p.grad.dtype, []).append(p.grad)
        for gs in buckets.values():
            flat = torch._utils._flatten_dense_tensors(gs); dist.all_reduce(flat); flat /= D.WORLD
            for gg, r in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
                gg.copy_(r)
        lt = torch.tensor([tot, tot_fm, tot_pen], device=dev); dist.all_reduce(lt)
        tot, tot_fm, tot_pen = (float(x) / D.WORLD for x in lt)
    for m_, p in zip(master, params):
        m_.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m_)
    gn = torch.nn.utils.clip_grad_norm_(master, 1.0)
    opt.step()
    with torch.no_grad():
        for m_, p in zip(master, params):
            p.copy_(m_.to(p.dtype))
    ema = tot if ema is None else 0.98 * ema + 0.02 * tot
    if D.IS_MAIN and step % 20 == 0:
        rec = {"step": step, "loss": tot, "l_fm": tot_fm, "l_pen": tot_pen, "ema": ema,
               "grad_norm": float(gn), "s_per_step": (time.time() - t0) / step, **st_last}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % 100 == 0:
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in rec.items()), flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        vm = run_val()
        # score: penetration down, drift bounded. All checkpoints are kept with
        # --keep-steps so the operating point can also be picked after the fact.
        score = vm["pen_tol"] + a.w_drift * vm["drift"]
        improved = score < best - 1e-5
        if improved:
            best, stalled = score, 0; save("denoiser_best.pt")
        else:
            stalled += 1
        save("denoiser_last.pt")
        if a.keep_steps:
            save(f"denoiser_step{step:07d}.pt")
        D.log(f"[val] step {step}  " + "  ".join(f"{k} {v:.4f}" for k, v in vm.items()) +
              f"  score {score:.5f} (best {best:.5f}, stalled {stalled}/{a.patience})  "
              f"{(time.time()-t0)/60:.1f} min")
        if D.IS_MAIN:
            logf.write(json.dumps({"step": step, "val": vm, "score": score, "best": improved}) + "\n"); logf.flush()
        if stalled >= a.patience:
            D.log("converged - stopping"); break
if D.IS_MAIN:
    if not (out / "ckpts" / "denoiser_best.pt").exists():
        save("denoiser_best.pt")
    (out / "DONE").write_text(f"{step} {(time.time()-t0)/60:.1f}min best {best}\n")
if D.DDP:
    dist.barrier(); dist.destroy_process_group()
