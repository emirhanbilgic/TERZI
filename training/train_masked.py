"""Large-scale fine-tune of the garment DiT (SS or SLat): masks + A-pose (full/, full_index.py).

Model = train_masked.py's (RGB + 4 body renders + a binary garment MASK as 6th DINOv3 view, zero-init
mask embedding), initialised from the ALIGNED v1 model (p=0; 5 view embeddings).

Every draw = (example, mode), chosen independently:
  example  uniform over ALL examples: per sample its whole outfit (mask = all garments) + each visible
           garment (mask + target = that garment) -> ~1/3 whole-outfit, as the formula
  mode     aligned    (--p-aligned 0.5): RGB, mask, body, target of the same frame
           misaligned (--p-mis 0.4):     RGB + its mask from ANOTHER frame of the same sequence (garment
                                         matched by its exact vertex count, visible there), body + target current
           A-pose     (rest, 0.1):       this frame's RGB + mask, the A-pose body, the A-pose garment(s)
  A mode only draws from the examples that can take it (no silent fallback to aligned), so the mode
  shares are exact; the realised (mode x whole/single) counts are logged.
Validation: fixed draws per cell (mode x whole/single), losses reported per cell.

  torchrun --nproc_per_node=<gpus> training/train_masked.py --kind ss   --out results/masked_ss   --init <main SS ckpt>
  torchrun --nproc_per_node=<gpus> training/train_masked.py --kind slat --out results/masked_slat --init <main SLat ckpt>
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)); _WD = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _WD, os.path.join(_WD, "TRELLIS.2")]
import ddp_setup as D   # FIRST: per-rank flex_gemm cache, set_device, seeds
import argparse, glob, io, json, math, pickle, random, tarfile, time, zlib
from pathlib import Path
import numpy as np, torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from trellis2 import models as _models
from trellis2.modules import sparse as sp
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from terzi import render as BR
from common import bake_cond

VIEWS = ("front", "back", "right", "left")
N_IMG = 1 + len(VIEWS) + 1          # RGB, 4 body views, garment mask

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["ss", "slat"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=None, help="default: config max_steps")
ap.add_argument("--save-every", type=int, default=None, help="default: config i_save")
ap.add_argument("--keep-every", type=int, default=10000, help="step ckpts kept permanently")
ap.add_argument("--val-every", type=int, default=None, help="default: config i_val")
ap.add_argument("--val-n", type=int, default=256)
ap.add_argument("--micro-bs", type=int, default=None, help="override config batch_size_per_gpu/batch_split")
ap.add_argument("--grad-accum", type=int, default=None, help="override config batch_split")
ap.add_argument("--body-token-pool", type=int, default=2,
                help="average-pool each body view's patch grid by this factor "
                     "(2: 32x32 -> 16x16 patches, KV 5145 -> 2073). 1 = no pooling")
ap.add_argument("--index", default="full/index.pkl", help="training/data/full_index.py output")
ap.add_argument("--data", default="two_branch_data", help="root of the prepared whole-outfit data (training/data/README.md)")
ap.add_argument("--p-aligned", type=float, default=0.5)
ap.add_argument("--p-mis", type=float, default=0.4, help="misaligned; A-pose gets the rest")
ap.add_argument("--mask-token-pool", type=int, default=1, help="pool the mask's patch grid (1 = full res, like the RGB)")
ap.add_argument("--init", required=True,
                help="the main model's checkpoint of --kind (train_main.py). A 5-view checkpoint gets a zero "
                     "mask embedding. Only used when not resuming.")
ap.add_argument("--lr", type=float, default=None, help="override the config's peak learning rate")
ap.add_argument("--warmup", type=int, default=None, help="override the config's warmup_steps")
ap.add_argument("--t-max", type=int, default=None, help="override the config's lr_scheduler T_max (decay horizon, steps)")
ap.add_argument("--smoke", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
KIND = {
    "slat": dict(config="configs/slat.json",
                 posed="shape/shape_latent", all="shape_garment_aligned/shape_latent", apose="shape_garment/shape_latent"),
    "ss": dict(config="configs/ss.json",
               posed="ss/ss_latent", all="ss_garment_aligned/ss_latent", apose="ss_garment/ss_latent"),
}[a.kind]
C = json.load(open(os.path.join(_WD, KIND["config"])))
tr = C["trainer"]["args"]
mcfg = C["models"]["denoiser"]
if a.kind == "ss":
    mcfg["args"]["in_channels"] = mcfg["args"]["out_channels"] = 8     # garment only
total = a.smoke or a.steps or int(tr["max_steps"])
save_every = a.save_every or int(tr.get("i_save", 1000))
val_every = a.val_every or int(tr.get("i_val", save_every))
bs = int(tr["batch_size_per_gpu"]); split = a.grad_accum or int(tr.get("batch_split", 1))
micro = a.micro_bs or (bs // int(tr.get("batch_split", 1)))
sigma_min = float(tr.get("sigma_min", 1e-5)); p_uncond = float(tr.get("p_uncond", 0.1))
tsch = tr.get("t_schedule", {"name": "uniform"})
out = Path(a.out)
state_p = out / "ckpts" / "train_state.pt"
RESUME = state_p.exists()
_rc = out / "run_config.json"
if D.IS_MAIN:
    # rank 0 alone decides (and writes), so a half-written run_config.json can
    # never be read by another rank: a non-empty dir is only reused when THIS
    # script created it for the same kind (a chained job restarting before its
    # first checkpoint); anything else belongs to another run and is untouched
    if out.exists() and any(out.iterdir()) and not RESUME:
        ok = False
        if _rc.exists():
            try:
                ok = json.load(open(_rc)).get("args", {}).get("kind") == a.kind
            except json.JSONDecodeError:
                ok = False
        if not ok:
            raise SystemExit(f"REFUSING: {out} is not an unfinished {a.kind} run of this script")
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    if not RESUME:
        tmp = str(_rc) + ".tmp"
        json.dump({"args": vars(a), "kind": KIND, "config": C, "views": VIEWS, "world": D.WORLD},
                  open(tmp, "w"), indent=1)
        os.replace(tmp, _rc)
if D.DDP:
    dist.barrier()
D.log(f"kind {a.kind} | out {out} | steps {total} | micro {micro} x accum {split} x {D.WORLD} GPUs "
      f"= global batch {micro * split * D.WORLD} (config global was {bs * 4}) "
      f"| t {tsch} | p_uncond {p_uncond} | resume {RESUME}")


# ---------------------------------------------------------------- model
class BodyCondDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = getattr(_models, mcfg["name"])(**mcfg["args"])
        self.view_embed = torch.nn.Parameter(torch.zeros(N_IMG, mcfg["args"]["cond_channels"]))

    def cond_tokens(self, rgb_tok, body_tok, mask_tok):
        """rgb_tok (B,L,C) full res + body_tok (B,4,L,C) pooled + mask_tok (B,L,C) ->
        (B, L + 4*L' + L'', C), each image tagged with its learned embedding."""
        e = self.view_embed.to(rgb_tok.dtype)
        body = pool_patches(body_tok, a.body_token_pool) + e[None, 1:N_IMG - 1, None, :]
        B, N, Lp, Cc = body.shape
        mask = pool_patches(mask_tok[:, None], a.mask_token_pool)[:, 0] + e[None, N_IMG - 1, None, :]
        return torch.cat([rgb_tok + e[None, 0, None, :], body.reshape(B, N * Lp, Cc), mask], 1)

    def forward(self, x, t, cond):
        return self.denoiser(x, t, cond)


def pool_patches(tok, f):
    """DINOv3 gives 1 CLS + 4 registers + 32x32 patch tokens. Average-pool the
    patch grid f x f (keeping the 5 non-patch tokens) -> fewer cross-attention keys."""
    if f <= 1:
        return tok
    B, N, L, Cc = tok.shape
    g = int(round((L - 5) ** 0.5))
    extra, patches = tok[:, :, :5], tok[:, :, 5:].reshape(B * N, g, g, Cc).permute(0, 3, 1, 2)
    p = F.avg_pool2d(patches, f).permute(0, 2, 3, 1).reshape(B, N, -1, Cc)
    return torch.cat([extra, p], 2)


model = BodyCondDenoiser()
if RESUME:
    st = torch.load(state_p, map_location="cpu")
    miss, unexp = model.load_state_dict({k: v for k, v in st["master"].items()}, strict=False)
    # the fp32 master holds parameters only; rope_phases is a buffer rebuilt at init
    assert not unexp and all(k.endswith("rope_phases") for k in miss), (miss, unexp)
else:
    src = a.init
    sd = torch.load(os.path.join(_WD, src) if not os.path.isabs(src) else src, map_location="cpu")
    if any(k.startswith("denoiser.") for k in sd):          # a checkpoint of this script or a 5-view one (v1)
        ve = sd["view_embed"]
        if ve.shape[0] == N_IMG - 1:                        # 5-view model: mask embedding starts at zero
            sd = dict(sd); sd["view_embed"] = torch.cat([ve, torch.zeros(1, ve.shape[1], dtype=ve.dtype)])
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert not unexp and all(k.endswith("rope_phases") for k in miss), (miss, unexp)
    else:                                                   # stock TRELLIS.2 denoiser weights
        miss, unexp = model.denoiser.load_state_dict(sd, strict=False)
    D.log(f"init {src}: missing {len(miss)} unexpected {len(unexp)}")
model = model.to(dev).train()
params = list(model.parameters()); names = [n for n, _ in model.named_parameters()]
master = [p.detach().float().clone().requires_grad_(True) for p in params]
if RESUME:
    for m, n in zip(master, names):
        m.data.copy_(st["master"][n].float())
ema_rate = float(tr.get("ema_rate", [0.9999])[0])
ema = [m.detach().clone() for m in master]
if RESUME:
    for e, n in zip(ema, names):
        e.copy_(st["ema"][n].float())
oargs = dict(tr["optimizer"]["args"])
if a.lr is not None:
    oargs["lr"] = a.lr
if a.warmup is not None:
    tr["lr_scheduler"]["args"]["warmup_steps"] = a.warmup
if a.t_max is not None:
    tr["lr_scheduler"]["args"]["T_max"] = a.t_max
opt = torch.optim.AdamW(master, **oargs)
if RESUME:
    opt.load_state_dict(st["opt"])
start = int(st["step"]) + 1 if RESUME else 1
clip_cfg = tr.get("grad_clip", 1.0)
adaptive = isinstance(clip_cfg, dict)
max_norm = float(clip_cfg["args"]["max_norm"]) if adaptive else float(clip_cfg)
gn_buf = list(st.get("gn_buf", [])) if RESUME else []
best_val = float(st.get("best_val", float("inf"))) if RESUME else float("inf")
sched = tr["lr_scheduler"]
D.log(f"params {sum(p.numel() for p in params)/1e9:.3f}B | opt {oargs} | sched {sched} | ema {ema_rate} | "
      f"clip {'adaptive p95 ' if adaptive else ''}{max_norm}")


def lr_at(step):
    """Closed forms of train_optimized.CosineAnnealingHold / ExponentialHold (last_epoch = step)."""
    base = oargs["lr"]; s = sched["args"]; T = int(s["T_max"]); eta = float(s["eta_min"])
    wu = int(s.get("warmup_steps", 0)); f0 = float(s.get("warmup_init_factor", 0.01))
    t = step
    if wu and t < wu:
        return base * (f0 + (1 - f0) * t / wu)
    u = min(t - wu, T - wu)
    if sched["name"] == "CosineAnnealingHold":
        return eta + (base - eta) * (1 + math.cos(math.pi * u / (T - wu))) / 2
    if sched["name"] == "ExponentialHold":
        return math.exp(math.log(base) + (u / (T - wu)) * (math.log(eta) - math.log(base)))
    raise ValueError(sched["name"])


# ---------------------------------------------------------------- frozen helpers
dino = DinoV3FeatureExtractor(tr["image_cond_model"]["args"]["model_name"], 512); dino.cuda()
body_dec = _models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(dev).eval()
body_dec.set_resolution(512)
for q in body_dec.parameters():
    q.requires_grad_(False)
if a.kind == "slat":
    norm = C["dataset"]["args"]["normalization"]
    MEAN = torch.tensor(norm["mean"], dtype=torch.float32)[None]; STD = torch.tensor(norm["std"], dtype=torch.float32)[None]


@torch.no_grad()
def cond_from(rgb, body_coords, body_feats, mask):
    """rgb (B,3,512,512) in [0,1]; body_* lists of numpy; mask (B,512,512) in {0,1}
    -> DINO feats (B, N_IMG, L, C) in the order RGB, 4 body views, mask."""
    B = rgb.shape[0]
    coords = torch.cat([torch.cat([torch.full((len(c), 1), i, dtype=torch.int32), torch.as_tensor(c).int()], 1)
                        for i, c in enumerate(body_coords)]).to(dev)
    feats = torch.cat([torch.as_tensor(f).float() for f in body_feats]).to(dev)
    meshes = body_dec(sp.SparseTensor(coords=coords, feats=feats))
    views = torch.stack([BR.render_views(m.vertices, m.faces, VIEWS) for m in meshes])   # B,4,3,S,S
    imgs = torch.cat([rgb[:, None], views, mask[:, None, None].expand(-1, 1, 3, -1, -1)], 1).reshape(B * N_IMG, 3, 512, 512)
    return dino(imgs).reshape(B, N_IMG, -1, dino_dim())


def dino_dim():
    return mcfg["args"]["cond_channels"]


# ---------------------------------------------------------------- data
HM = Path(_WD) / a.data
IX = pickle.load(open(a.index if os.path.isabs(a.index) else os.path.join(_WD, a.index), "rb"))
ROOTS = [dict(pg=Path(r["pg"]), ap=Path(r["ap"])) for r in IX["roots"]]
IDS = IX["ids"]; NG, NV, SINGLE, AP_ALL, AP_SINGLE = IX["ng"], IX["nverts"], IX["single"], IX["ap_all"], IX["ap_single"]
MODES = ("aligned", "mis", "apose")


def seq_members(train):
    """seq -> sample indices of that split (CSR)."""
    idx = np.nonzero(IX["split"] != train)[0]
    order = idx[np.argsort(IX["seq"][idx], kind="stable")]
    sq = IX["seq"][order]; starts = np.searchsorted(sq, np.arange(IX["seq"].max() + 2))
    return order, starts


def peer_match(s, g, q):
    """setting of sample q showing the same garment(s) as (s, g): "all", "g<j>" or None."""
    if g < 0:
        return "all"
    if NG[q] != NG[s] or NG[s] > len(NV[s]):
        return None
    j = np.nonzero(NV[q] == NV[s, g])[0]
    return f"g{j[0]}" if len(j) == 1 and SINGLE[q, j[0]] else None


def peers(s, g, members):
    order, starts = members; q = IX["seq"][s]
    return [int(x) for x in order[starts[q]:starts[q + 1]] if x != s and peer_match(s, g, int(x)) is not None]


def build(train):
    """examples (sample idx, garment idx or -1 = whole outfit) and, per mode, the examples that can take it."""
    sel = np.nonzero(IX["split"] != train)[0]
    ex_s = [sel]; ex_g = [np.full(len(sel), -1)]
    for k in range(SINGLE.shape[1]):
        m = sel[SINGLE[sel, k]]; ex_s.append(m); ex_g.append(np.full(len(m), k))
    ex_s = np.concatenate(ex_s).astype(np.int64); ex_g = np.concatenate(ex_g).astype(np.int8)
    members = seq_members(train)
    # misaligned-capable, vectorised: whole outfit -> another frame of the sequence exists; garment k ->
    # another frame with the same garment count where that vertex count is a unique, visible garment
    import pandas as pd
    seq = IX["seq"]; cnt = np.bincount(seq[sel], minlength=seq.max() + 1)
    r_s, r_k = np.nonzero(SINGLE[sel]); r_s = sel[r_s]
    uniq = np.array([(NV[q] == NV[q, k]).sum() == 1 for q, k in zip(r_s, r_k)], bool)
    t = pd.DataFrame(dict(seq=seq[r_s], ng=NG[r_s], nv=NV[r_s, r_k], u=uniq))
    t = t[t.u].groupby(["seq", "ng", "nv"]).size()
    e = pd.MultiIndex.from_arrays([seq[ex_s], NG[ex_s], NV[ex_s, np.maximum(ex_g, 0)]])
    others = t.reindex(e).fillna(0).to_numpy() - np.array([(NV[q] == NV[q, max(k, 0)]).sum() == 1 for q, k in zip(ex_s, ex_g)])
    has_mis = np.where(ex_g < 0, cnt[seq[ex_s]] > 1, others >= 1)
    has_ap = np.where(ex_g < 0, AP_ALL[ex_s], AP_SINGLE[ex_s, np.maximum(ex_g, 0)])
    pools = {"aligned": np.arange(len(ex_s)), "mis": np.nonzero(has_mis)[0], "apose": np.nonzero(has_ap)[0]}
    return ex_s, ex_g, pools, members


EX_S, EX_G, POOLS, MEMBERS = build(train=True)
D.log(f"train: {len(set(EX_S.tolist()))} samples, {len(EX_S)} examples ({(EX_G < 0).sum()} whole-outfit, {(EX_G >= 0).sum()} single) "
      f"| pools: " + ", ".join(f"{m} {len(p)} ({(EX_G[p] < 0).mean():.2f} whole)" for m, p in POOLS.items())
      + f" | p aligned {a.p_aligned} mis {a.p_mis} apose {1 - a.p_aligned - a.p_mis:.2f}")
_pm = {"aligned": a.p_aligned, "mis": a.p_mis, "apose": 1 - a.p_aligned - a.p_mis}
assert all(len(POOLS[m]) for m in MODES if _pm[m] > 0), "an empty mode pool"


def load_item(mode, s, g, view=None, gview=None):
    """(target, rgb, body coords, body feats, mask) for sample index s, garment g (-1 = whole outfit).
    view/gview (misaligned): RGB + mask of sample index `view`, its setting `gview`."""
    sid = IDS[s]; R = ROOTS[IX["root"][s]]
    view = s if view is None else view; vid = IDS[view]
    gview = gview or ("all" if g < 0 else f"g{g}")
    _, b = bake_cond(HM / "shape_garment_aligned/render_cond" / vid / "000.png", 512, "jpeg", 95)
    rgb = torch.from_numpy(np.array(Image.open(io.BytesIO(b)).convert("RGB"))).permute(2, 0, 1).float() / 255
    mask = torch.from_numpy(np.array(Image.open(ROOTS[IX["root"][view]]["pg"] / "masks" / vid / f"{gview}.png")) > 127).float()
    if mode == "apose":
        bd = np.load(R["ap"] / "shape_body/shape_latent" / f"{sid}__apose.npz")
        z = np.load(R["ap"] / KIND["apose"] / (f"{sid}__apose.npz" if g < 0 else f"{sid}__apose_g{g}.npz"))
    else:
        bd = np.load(HM / "shape_body_aligned/shape_latent" / f"{sid}.npz")
        z = np.load(HM / KIND["all"] / f"{sid}.npz") if g < 0 else np.load(R["pg"] / KIND["posed"] / f"{sid}__g{g}.npz")
    if a.kind == "slat":
        x = (torch.from_numpy(z["coords"].astype(np.int32)), (torch.from_numpy(z["feats"]).float() - MEAN) / STD)
    else:
        x = torch.from_numpy(z["z"]).float()
    return x, rgb, bd["coords"], bd["feats"], mask


def draw(rng, ex_s, ex_g, pools, members):
    """one (mode, s, g, view, gview) by the formula."""
    u = rng.random()
    mode = "aligned" if u < a.p_aligned else ("mis" if u < a.p_aligned + a.p_mis else "apose")
    i = pools[mode][rng.randrange(len(pools[mode]))]; s, g = int(ex_s[i]), int(ex_g[i])
    if mode != "mis":
        return mode, s, g, None, None
    q = rng.choice(peers(s, g, members))
    return mode, s, g, q, peer_match(s, g, q)


STATS = __import__("collections").Counter()


class Draws(torch.utils.data.IterableDataset):
    """endless draws by the formula; per rank x worker seeded."""
    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(1000 * D.RANK + (wi.id if wi else 0) + 7919 * int(time.time()))
        while True:
            mode, s, g, view, gview = draw(rng, EX_S, EX_G, POOLS, MEMBERS)
            STATS[f"{mode}/{'whole' if g < 0 else 'single'}"] += 1; STATS["n"] += 1
            if STATS["n"] % 20000 == 0:
                n = STATS["n"]
                print(f"[draws w{wi.id if wi else 0} r{D.RANK}] n {n} | " + " ".join(
                    f"{k} {v / n:.3f}" for k, v in sorted(STATS.items()) if k != "n"), flush=True)
            try:
                yield load_item(mode, s, g, view, gview)
            except Exception as e:                      # a corrupt file must not stop training
                print(f"[skip] {mode} {IDS[s]} {g}: {e}", flush=True)


loader = iter(torch.utils.data.DataLoader(Draws(), batch_size=None, num_workers=int(os.environ.get("GBC_WORKERS", "8")),
                                          prefetch_factor=4, persistent_workers=True))


def collate(items):
    rgb = torch.stack([it[1] for it in items]).to(dev, non_blocking=True)
    bc = [it[2] for it in items]; bf = [it[3] for it in items]
    mask = torch.stack([it[4] for it in items]).to(dev, non_blocking=True)
    if a.kind == "slat":
        coords = torch.cat([torch.cat([torch.full((len(it[0][0]), 1), i, dtype=torch.int32), it[0][0]], 1)
                            for i, it in enumerate(items)]).to(dev)
        x0 = sp.SparseTensor(coords=coords, feats=torch.cat([it[0][1] for it in items]).to(dev))
    else:
        x0 = torch.stack([it[0] for it in items]).to(dev)
    return x0, rgb, bc, bf, mask


def sample_t(n, g=None):
    if tsch["name"] == "logitNormal":
        return torch.sigmoid(torch.randn(n, generator=g) * tsch["args"]["std"] + tsch["args"]["mean"])
    return torch.rand(n, generator=g)


def fm_loss(x0, feats, drop, t, eps=None):
    """Flow-matching MSE exactly as the trainer (SLat: mean over all tokens of the batch)."""
    cond = model.cond_tokens(feats[:, 0], feats[:, 1:N_IMG - 1], feats[:, N_IMG - 1])
    cond = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)
    if a.kind == "slat":
        e = torch.randn_like(x0.feats) if eps is None else eps
        tt = t[x0.coords[:, 0].long()][:, None]
        xt = x0.replace((1 - tt) * x0.feats + (sigma_min + (1 - sigma_min) * tt) * e)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xt, t * 1000, cond).feats
        return F.mse_loss(pred.float(), ((1 - sigma_min) * e - x0.feats).float())
    e = torch.randn_like(x0) if eps is None else eps
    tt = t.view(-1, 1, 1, 1, 1)
    xt = (1 - tt) * x0 + (sigma_min + (1 - sigma_min) * tt) * e
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(xt, t * 1000, cond)
    return F.mse_loss(pred.float(), ((1 - sigma_min) * e - x0).float())


# ---------------------------------------------------------------- validation (fixed draws)
CELLS = [f"{m}/{w}" for m in MODES for w in ("whole", "single")]
VAL = []          # (cell, mode, s, g, view, gview): val_n / 6 fixed draws per cell
if D.IS_MAIN:
    vs, vg, vpools, vmem = build(train=False)
    rng_v = random.Random(0)
    for m in MODES:
        for w in ("whole", "single"):
            p = [i for i in vpools[m] if (vg[i] < 0) == (w == "whole")]
            for i in rng_v.sample(p, min(len(p), a.val_n // len(CELLS))):
                s_, g_ = int(vs[i]), int(vg[i])
                if m == "mis":
                    q = sorted(peers(s_, g_, vmem))[0]; VAL.append((f"{m}/{w}", m, s_, g_, q, peer_match(s_, g_, q)))
                else:
                    VAL.append((f"{m}/{w}", m, s_, g_, None, None))
    D.log("val draws: " + ", ".join(f"{c} {sum(v[0] == c for v in VAL)}" for c in CELLS))


@torch.no_grad()
def run_val():
    model.eval(); losses = {c: [] for c in CELLS}
    for i in range(0, len(VAL), micro):
        chunk = VAL[i:i + micro]
        x0, rgb, bc, bf, mask = collate([load_item(*v[1:]) for v in chunk])
        feats = cond_from(rgb, bc, bf, mask)
        key = ",".join(f"{v[1]}/{IDS[v[2]]}/{v[3]}" for v in chunk)
        g = torch.Generator().manual_seed(zlib.crc32(key.encode()))
        t = sample_t(len(chunk), g).to(dev)
        eps_shape = x0.feats.shape if a.kind == "slat" else x0.shape
        eps = torch.randn(eps_shape, generator=g).to(dev)
        drop = torch.zeros(len(chunk), dtype=torch.bool, device=dev)     # conditional loss
        for j, v in enumerate(chunk):                                     # per item, so cells separate
            if a.kind == "slat":
                sel = x0.coords[:, 0] == j
                xj = sp.SparseTensor(coords=torch.cat([torch.zeros_like(x0.coords[sel][:, :1]), x0.coords[sel][:, 1:]], 1),
                                     feats=x0.feats[sel])
                lj = fm_loss(xj, feats[j:j + 1], drop[j:j + 1], t[j:j + 1], eps[sel])
            else:
                lj = fm_loss(x0[j:j + 1], feats[j:j + 1], drop[j:j + 1], t[j:j + 1], eps[j:j + 1])
            losses[v[0]].append(float(lj))
    model.train()
    m = {c: float(np.mean(v)) if v else float("nan") for c, v in losses.items()}
    m["mean"] = float(np.mean(sum(losses.values(), [])))
    return m


# ---------------------------------------------------------------- train
def save(step, tagged):
    cast = lambda ts: {n: x.detach().to(q.dtype).cpu() for n, x, q in zip(names, ts, params)}
    if tagged:
        torch.save(cast(master), out / "ckpts" / f"denoiser_step{step:07d}.pt")
        torch.save(cast(ema), out / "ckpts" / f"denoiser_ema_step{step:07d}.pt")
    torch.save(cast(master), out / "ckpts" / "denoiser_last.pt.tmp"); os.replace(out / "ckpts" / "denoiser_last.pt.tmp", out / "ckpts" / "denoiser_last.pt")
    torch.save({"step": step, "opt": opt.state_dict(), "gn_buf": gn_buf, "best_val": best_val,
                "master": {n: m.detach().float().cpu() for n, m in zip(names, master)},
                "ema": {n: e.detach().float().cpu() for n, e in zip(names, ema)}}, str(state_p) + ".tmp")
    os.replace(str(state_p) + ".tmp", state_p)


logf = open(out / "log.jsonl", "a") if D.IS_MAIN else None
t_start, lema = time.time(), None
for step in range(start, total + 1):
    lr = lr_at(step)
    for grp in opt.param_groups:
        grp["lr"] = lr
    for p in params:
        p.grad = None
    lacc = 0.0; t_data = t_cond = 0.0
    for _ in range(split):
        t0 = time.time(); x0, rgb, bc, bf, mask = collate([next(loader) for _ in range(micro)]); t_data += time.time() - t0
        t0 = time.time(); feats = cond_from(rgb, bc, bf, mask); t_cond += time.time() - t0
        drop = torch.from_numpy(np.random.rand(micro) < p_uncond).to(dev)
        t = sample_t(micro).to(dev)
        loss = fm_loss(x0, feats, drop, t)
        (loss / split).backward()
        lacc += float(loss.detach()) / split
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
    for m, p in zip(master, params):
        m.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m)
    cur_max = min(float(np.percentile(gn_buf[-1000:], 95)), max_norm) if adaptive and len(gn_buf) >= 1000 else max_norm
    gn = torch.nn.utils.clip_grad_norm_(master, cur_max)
    if torch.isfinite(gn):
        gn_buf.append(float(gn)); gn_buf = gn_buf[-1000:]
    opt.step()
    with torch.no_grad():
        for m, p in zip(master, params):
            p.copy_(m.to(p.dtype))
        for e, m in zip(ema, master):
            e.mul_(ema_rate).add_(m.detach(), alpha=1 - ema_rate)
    lema = lacc if lema is None else 0.98 * lema + 0.02 * lacc
    if D.IS_MAIN and (step % 10 == 0 or step <= 3):
        rec = {"step": step, "loss": lacc, "loss_ema": lema, "lr": lr, "grad_norm": float(gn), "clip": cur_max,
               "s_per_step": (time.time() - t_start) / (step - start + 1), "t_data": t_data, "t_cond": t_cond}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % 50 == 0 or step <= 3:
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
    if D.IS_MAIN and VAL and (step % val_every == 0 or step == total):
        vm = run_val(); vl = vm["mean"]
        is_best = vl < best_val - 1e-6
        if is_best:
            best_val = vl
        rec = {"step": step, "val_loss": vl, **{f"val_{c}": vm[c] for c in CELLS}, "best": is_best}
        logf.write(json.dumps(rec) + "\n"); logf.flush(); print(f"  [val] {rec}", flush=True)
        if is_best:
            cast = lambda ts: {n: x.detach().to(q.dtype).cpu() for n, x, q in zip(names, ts, params)}
            torch.save(cast(master), out / "ckpts" / "denoiser_best.pt"); torch.save(cast(ema), out / "ckpts" / "denoiser_ema_best.pt")
    if D.IS_MAIN and (step % save_every == 0 or step == total):
        save(step, tagged=(step % a.keep_every == 0 or step == total))
        print(f"  saved step {step}", flush=True)
    if D.DDP and step % 50 == 0:
        dist.barrier()
if D.IS_MAIN and not a.smoke:
    (out / "DONE").write_text(f"{total}\n")
if D.DDP:
    dist.barrier(); dist.destroy_process_group()
