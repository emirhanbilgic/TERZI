#!/usr/bin/env python3
"""LAYER fine-tune of the per-garment (mask-conditioned, 6-view) garment SLat DiT.

= finetune_main.py (the FT-pen recipe) with a second, garment-garment penalty:

  x_t   = (1-t) z_gen + (sigma + (1-sigma) t) eps         z_gen, eps: the FROZEN model's own sample (datagen_masked)
  L_fm  = || v - v_frozen(x_t) ||^2                        self-anchor (frozen copy of --init)
  x0hat = x_t - t v ;  mesh = decode(x0hat)                one-step estimate, differentiable vertices
  L_body  = hinge(clearance - sdf_body(mesh))              pen_loss, watertight GT SMPL-X
  L_layer = sum over the OTHER garments G of this sample that overlap this one (GT order):
              this garment OUTER of G: hinge(clearance - s_G(mesh))
              this garment INNER of G: hinge(clearance + s_G(mesh))
            s_G = local signed distance along G's outward normal (layer_loss), G = the frozen model's
            own generation of that garment (what gets stacked at inference)
  L     = w_anchor L_fm + w_pen L_body + w_layer L_layer   penalties only for t <= pen_tmax, conditional steps

"Both directions": each garment is pushed to its correct side of every overlapping layer in
its own steps (outer outwards, inner inwards). ALL-mask draws (--p-all) get anchor + body only.
SS is untouched (coords come from the frozen SS, as in finetune_main).

Validation (every --eval-every): every single garment of --val-n held-out samples is re-generated
with the CURRENT model (stored coords + noise, 12-step CFG) and stacked: layer crossing of each
ordered pair (both garments current), body penetration, drift vs the frozen generation.
score = body pen_tol + cross + w_drift * drift  -> denoiser_best.pt.

Paper setting (TERZI (masked), supplementary Tab. 6): the layer penalty is OFF (--w-layer 0, the default here), so this is the
penetration fine-tune of the masked model; the released checkpoint is step 100.
  torchrun --nproc_per_node=1 training/finetune/finetune_masked.py --data <datagen_masked.py output>/train \
      --val-data <...>/val --init <masked SLat ckpt> --out results/ft_masked --keep-steps
"""
import sys, os, json, time, argparse, copy
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
_GBC = os.path.dirname(_HERE); _WD = os.path.dirname(_GBC)
sys.path[:0] = [_HERE, _GBC, _WD,
                os.path.join(_WD, "TRELLIS.2")]
import ddp_setup as D   # FIRST
import numpy as np, torch
import torch.distributed as dist
from trellis2 import models as _models
from trellis2.modules import sparse as sp
from rollout import rollout_x0                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="datagen_masked.py out dir(s), comma-separated (train)")
ap.add_argument("--val-data", required=True, help="datagen_masked.py --split val out dir")
ap.add_argument("--init", required=True, help="SLat checkpoint of the masked model (train_masked.py, 6 views)")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--w-anchor", type=float, default=1.0)
ap.add_argument("--w-pen", type=float, default=30.0, help="body penetration weight (as finetune_main.py)")
ap.add_argument("--w-layer", type=float, default=0.0, help="garment-garment layer penalty (0 in the paper)")
ap.add_argument("--p-all", type=float, default=0.2, help="draws of the ALL setting (anchor + body only)")
ap.add_argument("--p-uncond", type=float, default=0.1)
ap.add_argument("--clearance", type=float, default=1.5, help="crop px, body AND layers (1.5 px ~ 5.7 mm, paper)")
ap.add_argument("--deep-clip", type=float, default=2.0)
ap.add_argument("--r-max", type=float, default=8.0, help="px: a layer only counts within this distance of G")
ap.add_argument("--pen-tmax", type=float, default=0.6)
ap.add_argument("--w-drift", type=float, default=0.2)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--val-n", type=int, default=48)
ap.add_argument("--val-steps", type=int, default=12)
ap.add_argument("--patience", type=int, default=99)
ap.add_argument("--keep-steps", action="store_true")
ap.add_argument("--sequential", action="store_true",
                help="step-by-step generation: garments ordered outermost first (GT layer order); each garment is "
                     "penalised only against the garments generated BEFORE it (jacket -> shirt -> trousers)")
ap.add_argument("--max-cells", type=int, default=1200)
ap.add_argument("--pen-rollout", type=int, default=1,
                help="Euler steps from t to 0 for the penalised clean estimate (1 = x_t - t v; CF/rollout.py)")
ap.add_argument("--pen-rollout-grad", choices=["all", "last"], default="all")
ap.add_argument("--r-look", type=float, default=12.0, help="px: farther from any GT overlap point -> the pair's majority order")
ap.add_argument("--order-conf", type=float, default=0.3, help="local GT share in (conf, 1-conf) = tuck line, unconstrained")
ap.add_argument("--body-clearance", type=float, default=None,
                help="crop px for the BODY term only (default = --clearance); 0.4 px ~ 1.6 mm as the original FT-pen")
ap.add_argument("--ram-cap", type=int, default=6000, help="samples kept in host RAM")
a = ap.parse_args()
dev = "cuda"; SIGMA = 1e-5
BODY_CLR = a.body_clearance if a.body_clearance is not None else a.clearance
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
from common import to_metric                                           # noqa: E402
from pen_loss import PenaltyBody, penetration_penalty, surface_vertex_mask, fast_bvh  # noqa: E402
from layer_loss import LayerSurface, layer_penalty, LocalOrder                       # noqa: E402


class BodyCondDenoiser(torch.nn.Module):
    """train_masked.BodyCondDenoiser (same state-dict keys); cond from cached tokens."""
    def __init__(self):
        super().__init__()
        self.denoiser = getattr(_models, mcfg["name"])(**dict(mcfg["args"]))
        self.view_embed = torch.nn.Parameter(torch.zeros(6, mcfg["args"]["cond_channels"]))

    def cond(self, tok_rgb, tok_body, tok_mask):
        e = self.view_embed.to(tok_rgb.dtype)
        body = tok_body + e[1:5, None, :]
        return torch.cat([tok_rgb + e[0], body.reshape(-1, body.shape[-1]), tok_mask + e[5]], 0)[None]


model = BodyCondDenoiser()
sd = torch.load(a.init, map_location="cpu")
miss, unexp = model.load_state_dict(sd, strict=False)
assert set(m.split(".")[-1] for m in miss) <= {"rope_phases"} and not unexp, (miss, unexp)
model = model.to(dev).train()
params = list(model.parameters()); names = [n for n, _ in model.named_parameters()]
master = [p.detach().float().clone().requires_grad_(True) for p in params]
opt = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.0)
model0 = copy.deepcopy(model).eval()
for q in model0.parameters():
    q.requires_grad_(False)
dec = _models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(dev).eval()
dec.set_resolution(512)
D.log(f"init {a.init} | world {D.WORLD} x accum {a.grad_accum} | lr {a.lr} | w_pen {a.w_pen} w_layer {a.w_layer} "
      f"| p_all {a.p_all} | clearance {a.clearance} px | r_max {a.r_max} px")

from trellis2.pipelines import samplers                                 # noqa: E402
from huggingface_hub import hf_hub_download                             # noqa: E402
pj = json.load(open(hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")))["args"]
SAMPLER = getattr(samplers, pj["shape_slat_sampler"]["name"])(**pj["shape_slat_sampler"]["args"])
SPARAMS = dict(pj["shape_slat_sampler"]["params"]); SPARAMS["steps"] = a.val_steps


# ------------------------------------------------------------------ data
def index(root):
    r = Path(root); out = {}
    for d in sorted(r.iterdir()) if r.exists() else []:
        if (d / "layers.json").exists():
            L = json.load(open(d / "layers.json"))
            out[d.name] = dict(root=d, settings=L["settings"], pairs=L["pairs"], lpairs=L.get("local_pairs", []))
    return out


TRAIN = {}
for r in a.data.split(","):
    TRAIN.update(index(r))
VALD = index(a.val_data)
VAL = sorted(VALD)[:a.val_n]
TRAIN_IDS = sorted(TRAIN)
D.log(f"train samples {len(TRAIN_IDS)} ({sum(len(v['settings']) - 1 for v in TRAIN.values())} single garments, "
      f"{sum(len(v['pairs']) for v in TRAIN.values())} ordered layer pairs) | val samples {len(VAL)}")
assert len(TRAIN_IDS) >= 4, "not enough datagen_masked samples"
RAM = {}


def item(s, meta):
    if s in RAM:
        return RAM[s]
    d = meta["root"]; cz = np.load(d / "cond.npz"); bz = np.load(d / "body.npz")
    e = {"tok_rgb": torch.as_tensor(cz["tok_rgb"]), "tok_body": torch.as_tensor(cz["tok_body"]),
         "mask": {g: torch.as_tensor(cz[f"tok_mask_{g}"]) for g in meta["settings"]},
         "bv": torch.as_tensor(bz["body_cube_v"]).float(), "bf": torch.as_tensor(bz["body_cube_f"].astype(np.int64)),
         "lat": {}, "order": {}}
    if (d / "order.npz").exists():
        oz = np.load(d / "order.npz")
        for gi, gj, _ in meta["lpairs"]:
            k = f"{gi}|{gj}"; e["order"][(gi, gj)] = (torch.as_tensor(oz[k + "_pts"]), torch.as_tensor(oz[k + "_f"]), float(oz[k + "_maj"]))
    for g in meta["settings"]:
        z = np.load(d / f"lat_{g}.npz")
        e["lat"][g] = {"coords": torch.as_tensor(z["coords"].astype(np.int32)),
                       "z": torch.as_tensor((z["feats"].astype(np.float32) - MEAN.cpu().numpy()) / STD.cpu().numpy()).half(),
                       "eps": torch.as_tensor(z["eps"])}
    if len(RAM) >= a.ram_cap:
        RAM.pop(next(iter(RAM)))
    RAM[s] = e
    return e


def pack(c):
    return torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), c.int()], 1).to(dev)


GPU_CAP = 256
BODY, SURF, FLIP = {}, {}, [None]


def _cap(dct):
    if len(dct) >= GPU_CAP:
        dct.pop(next(iter(dct)))


def body_of(s, e):
    if s not in BODY:
        raw, FLIP[0] = fast_bvh(to_metric(e["bv"].to(dev)), e["bf"].to(dev), FLIP[0])
        _cap(BODY); BODY[s] = PenaltyBody(raw)
    return BODY[s]


@torch.no_grad()
def decode(coords, z_norm):
    m = dec(sp.SparseTensor(coords=coords, feats=z_norm.float() * STD + MEAN))[0]
    vv = to_metric(m.vertices.float())
    return vv, m.faces.long(), surface_vertex_mask(m.faces.long(), vv.shape[0], dev)


def frozen_surface(s, g, e):
    """LayerSurface of the FROZEN model's own generation of garment g."""
    k = (s, g)
    if k not in SURF:
        L = e["lat"][g]; vv, f, _ = decode(pack(L["coords"]), L["z"].to(dev))
        _cap(SURF); SURF[k] = LayerSurface(vv, f, body_of(s, e), r_max=a.r_max)
    return SURF[k]


def seq_order(meta):
    """garments ordered outermost first: score = (#pairs where outer) - (#pairs where inner), ties by name"""
    gs = sorted({x for i, o, _ in meta["pairs"] for x in (i, o)})
    sc = {x: sum(o == x for _, o, _ in meta["pairs"]) - sum(i == x for i, _, _ in meta["pairs"]) for x in gs}
    return sorted(gs, key=lambda x: (-sc[x], x))


def local_partners(meta, e, g):
    """[(other garment, LocalOrder, g is the pair's first garment)] for every GT-overlapping pair containing g"""
    P = []
    for (gi, gj), (pts, f, maj) in e["order"].items():
        if g in (gi, gj):
            P.append((gj if g == gi else gi, LocalOrder(pts.to(dev), f.to(dev), maj, r_look=a.r_look, conf=a.order_conf), g == gi))
    return P


def partners(meta, g):
    """[(other garment, this one is OUTER)] for the GT-ordered pairs containing g. With --sequential only the
    garments that come BEFORE g in the outermost-first sequence (already generated) count."""
    P = [(i, True) for i, o, _ in meta["pairs"] if o == g] + [(o, False) for i, o, _ in meta["pairs"] if i == g]
    if a.sequential:
        seq = seq_order(meta); k = seq.index(g) if g in seq else -1
        P = [(x, out) for x, out in P if x in seq and seq.index(x) < k]
    return P


# ------------------------------------------------------------------ validation
MYVAL = VAL[D.RANK::D.WORLD]
REF = {}


def _chamfer(a_, b_, n=8000, chunk=256):
    g = torch.Generator(device=dev).manual_seed(0)

    def half(X, Y):
        X = X[torch.randperm(len(X), device=dev, generator=g)[:n]]
        o = 0.0
        for i in range(0, len(X), chunk):
            o += float(torch.cdist(X[i:i + chunk], Y).min(1).values.sum())
        return o / max(len(X), 1)
    return 0.5 * (half(a_, b_) + half(b_, a_))


@torch.no_grad()
def run_val(frozen=False):
    """frozen=True: the stored frozen generations themselves (the step-0 reference)."""
    model.eval()
    acc = {"pen": 0.0, "pen_tol": 0.0, "cross": 0.0, "cross_outer": 0.0, "cross_inner": 0.0, "drift": 0.0}
    cnt = {"g": 0, "p": 0}
    for s in MYVAL:
        meta = VALD[s]; e = item(s, meta); body = body_of(s, e); meshes = {}
        for g in meta["settings"]:
            if g == "all":
                continue
            L = e["lat"][g]; coords = pack(L["coords"])
            if s + g not in REF:
                REF[s + g] = decode(coords, L["z"].to(dev))
            if frozen:
                vv, f, mk = REF[s + g]
            else:
                cond = model.cond(e["tok_rgb"].to(dev), e["tok_body"].to(dev), e["mask"][g].to(dev))
                noise = sp.SparseTensor(feats=L["eps"].to(dev).float(), coords=coords)
                z = SAMPLER.sample(model.denoiser, noise, cond=cond, neg_cond=torch.zeros_like(cond), **SPARAMS, verbose=False).samples.feats
                vv, f, mk = decode(coords, z)
            meshes[g] = (vv, f, mk)
            sd_ = body.bb.signed(vv[mk])
            acc["pen"] += float((sd_ < 0).float().mean()); acc["pen_tol"] += float((sd_ < -0.05 / 3.79).float().mean())
            acc["drift"] += 0.0 if frozen else _chamfer(vv[mk], REF[s + g][0][REF[s + g][2]])
            cnt["g"] += 1
        VP = ([(gi, gj, LocalOrder(pts.to(dev), f.to(dev), maj, r_look=a.r_look, conf=a.order_conf)) for (gi, gj), (pts, f, maj) in e["order"].items()]
              if e["order"] else [(gi, go, None) for gi, go, _ in meta["pairs"]])
        for gi, go, lo in VP:                  # local order: go is simply the pair's second garment
            si = LayerSurface(meshes[gi][0], meshes[gi][1], body, r_max=a.r_max)
            so = LayerSurface(meshes[go][0], meshes[go][1], body, r_max=a.r_max)
            qo, qi = meshes[go][0][meshes[go][2]], meshes[gi][0][meshes[gi][2]]
            _, sto = layer_penalty(qo, si, outer=True if lo is None else lo.want(qo, False))
            _, sti = layer_penalty(qi, so, outer=False if lo is None else lo.want(qi, True))
            acc["cross_outer"] += sto["cross"]; acc["cross_inner"] += sti["cross"]
            acc["cross"] += 0.5 * (sto["cross"] + sti["cross"]); cnt["p"] += 1
    model.train()
    t = torch.tensor(list(acc.values()) + [cnt["g"], cnt["p"]], device=dev, dtype=torch.float64)
    if D.DDP:
        dist.all_reduce(t)
    ng, npair = max(float(t[-2]), 1), max(float(t[-1]), 1)
    r = {k: float(t[i]) / (npair if k.startswith("cross") else ng) for i, k in enumerate(acc)}
    r["n_garments"], r["n_pairs"] = int(t[-2]), int(t[-1])
    return r


def save_state(step, best, stalled):
    if D.IS_MAIN:
        p = out / "ckpts" / "train_state.pt"
        torch.save({"step": step, "best": best, "stalled": stalled, "opt": opt.state_dict(),
                    "master": [m.detach().cpu() for m in master]}, str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)


def save(name):
    if D.IS_MAIN:
        p = out / "ckpts" / name
        torch.save({n: m.detach().to(q.dtype).cpu() for n, m, q in zip(names, master, params)}, str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)


# ------------------------------------------------------------------ train
logf = open(out / "log.jsonl", "a") if D.IS_MAIN else None
START, RES = 0, None
if (out / "ckpts" / "train_state.pt").exists():                       # resume (8 h interactive slots)
    RES = torch.load(out / "ckpts" / "train_state.pt", map_location="cpu")
    with torch.no_grad():
        for m_, x, p in zip(master, RES["master"], params):
            m_.copy_(x.to(dev)); p.copy_(m_.to(p.dtype))
    opt.load_state_dict(RES["opt"]); START = int(RES["step"])
    D.log(f"resumed at step {START} (best {RES['best']:.5f})")
r0 = run_val(frozen=True) if RES is None else {}
D.log("[ref] frozen generations on the val set: " + "  ".join(f"{k} {v:.4f}" for k, v in r0.items()))
v0 = run_val() if RES is None else {}
if RES is None: D.log("[val] step 0: " + "  ".join(f"{k} {v:.4f}" for k, v in v0.items()))
if D.IS_MAIN and RES is None:
    logf.write(json.dumps({"step": 0, "ref": r0, "val": v0}) + "\n"); logf.flush()

rng = np.random.default_rng(1000 + D.RANK)
order, best, stalled, ema, t0 = [], float("inf"), 0, None, time.time()
if RES is not None:
    best, stalled = RES["best"], RES["stalled"]
step = START
for step in range(START + 1, a.steps + 1):
    for p in params:
        p.grad = None
    tot = tot_fm = tot_pen = tot_lay = 0.0; st_last = {}
    for _ in range(a.grad_accum):
        if not order:
            order = list(rng.permutation(len(TRAIN_IDS)))
        s = TRAIN_IDS[order.pop()]; meta = TRAIN[s]; e = item(s, meta)
        singles = [g for g in meta["settings"] if g != "all"]
        g = "all" if rng.uniform() < a.p_all and "all" in meta["settings"] else singles[rng.integers(len(singles))]
        L = e["lat"][g]
        if len(L["coords"]) > a.max_cells:
            continue
        coords = pack(L["coords"]); z = L["z"].to(dev).float(); eps = L["eps"].to(dev).float()
        cond = model.cond(e["tok_rgb"].to(dev), e["tok_body"].to(dev), e["mask"][g].to(dev))
        uncond_step = bool(rng.uniform() < a.p_uncond)
        if uncond_step:
            cond = torch.zeros_like(cond)
        t = float(rng.uniform(0.02, 0.98))
        x_t = (1 - t) * z + (SIGMA + (1 - SIGMA) * t) * eps
        tt = torch.full((1,), 1000.0 * t, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            target = model0.denoiser(sp.SparseTensor(coords=coords, feats=x_t), tt, cond).feats.float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model.denoiser(sp.SparseTensor(coords=coords, feats=x_t), tt, cond).feats
        v = v.float()
        l_fm = ((v - target) ** 2).mean()
        loss = a.w_anchor * l_fm
        l_pen = l_lay = torch.zeros((), device=dev)
        if t <= a.pen_tmax and not uncond_step and (a.w_pen > 0 or a.w_layer > 0):
            if a.pen_rollout > 1:
                def _vel(f_, tk_):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return model.denoiser(sp.SparseTensor(coords=coords, feats=f_), torch.full((1,), 1000.0 * tk_, device=dev), cond).feats.float()
                zhat = rollout_x0(_vel, x_t, t, v, a.pen_rollout, grad=a.pen_rollout_grad)
            else:
                zhat = x_t - t * v
            mesh = dec(sp.SparseTensor(coords=coords, feats=zhat * STD + MEAN))[0]
            vv = to_metric(mesh.vertices.float())
            mk = surface_vertex_mask(mesh.faces.long(), vv.shape[0], dev)
            body = body_of(s, e)
            if a.w_pen > 0:
                l_pen, st_b = penetration_penalty(vv[mk], body, clearance=BODY_CLR, deep_clip=a.deep_clip)
                st_last.update({"body_in": st_b["frac_in"]})
                loss = loss + a.w_pen * l_pen
            if a.w_layer > 0 and g != "all":
                PP = ([(o, lo.want(vv[mk], fi)) for o, lo, fi in local_partners(meta, e, g)] if e["order"] and not a.sequential
                      else partners(meta, g))
                for other, is_outer in PP:
                    lp, st_l = layer_penalty(vv[mk], frozen_surface(s, other, e), outer=is_outer,
                                             clearance=a.clearance, deep_clip=a.deep_clip)
                    l_lay = l_lay + lp
                    st_last.update({"layer_cross": st_l["cross"], "layer_overlap": st_l["overlap"]})
                loss = loss + a.w_layer * l_lay
        (loss / a.grad_accum).backward()
        tot += float(loss) / a.grad_accum; tot_fm += float(l_fm) / a.grad_accum
        tot_pen += float(l_pen) / a.grad_accum; tot_lay += float(l_lay) / a.grad_accum
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
        lt = torch.tensor([tot, tot_fm, tot_pen, tot_lay], device=dev); dist.all_reduce(lt)
        tot, tot_fm, tot_pen, tot_lay = (float(x) / D.WORLD for x in lt)
    for m_, p in zip(master, params):
        m_.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m_)
    gn = torch.nn.utils.clip_grad_norm_(master, 1.0)
    opt.step()
    with torch.no_grad():
        for m_, p in zip(master, params):
            p.copy_(m_.to(p.dtype))
    ema = tot if ema is None else 0.98 * ema + 0.02 * tot
    if D.IS_MAIN and step % 20 == 0:
        rec = {"step": step, "loss": tot, "l_fm": tot_fm, "l_pen": tot_pen, "l_layer": tot_lay, "ema": ema,
               "grad_norm": float(gn), "s_per_step": (time.time() - t0) / max(step - START, 1), **st_last}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % 100 == 0:
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        vm = run_val()
        score = vm["pen_tol"] + vm["cross"] + a.w_drift * vm["drift"]
        improved = score < best - 1e-5
        if improved:
            best, stalled = score, 0; save("denoiser_best.pt")
        else:
            stalled += 1
        save("denoiser_last.pt"); save_state(step, best, stalled)
        if a.keep_steps:
            save(f"denoiser_step{step:07d}.pt")
        D.log(f"[val] step {step}  " + "  ".join(f"{k} {v:.4f}" for k, v in vm.items()) +
              f"  score {score:.5f} (best {best:.5f}, stalled {stalled}/{a.patience})  {(time.time()-t0)/60:.1f} min")
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
