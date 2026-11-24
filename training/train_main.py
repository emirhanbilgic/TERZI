"""Garment-only SS / SLat DiT conditioned on the RGB crop + 4-view GT-body renders.

  input  : RGB crop (the baked cond.jpg of the shard, as before)
           + GT body rendered on the fly (body_render.py) from front/back/right/left,
             image-aligned with the crop, as normal maps
  output : garment only
           ss   -> 8-channel 16^3 sparse-structure latent (zgarment), stock TRELLIS.2 layout
           slat -> garment structured latent (coords/feats of the garment shard)

Conditioning tokens: every image (1 RGB + 4 body views) goes through the same frozen
DINOv3 (1029 tokens x 1024 each); a learned per-image embedding (zero-init) is added
and the 5 token sets are concatenated along the sequence axis -> cross-attention KV
of 5145 tokens. CFG training drops the WHOLE conditioning to zeros with p_uncond, so
the inference negative is zeros_like(cond) exactly as in the stock pipeline.

The GT body comes from the body shard of the same sha: its SC-VAE latent is decoded
with the frozen shape decoder (512) on the GPU and rendered with nvdiffrast. Nothing
is written to disk. Samples whose body latent is missing are skipped.

Recipe = the original trainer's config (optimizer, lr_scheduler closed form incl.
warmup_init_factor 0.01, batch_size_per_gpu / batch_split, t_schedule, p_uncond,
sigma_min, grad clip, EMA), initialised from the same stock checkpoints. One
deliberate change: fp32 master weights.

  torchrun --nproc_per_node=<gpus> training/train_main.py --kind ss   --out results/main_ss
  torchrun --nproc_per_node=<gpus> training/train_main.py --kind slat --out results/main_slat
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
import webdataset as wds

from trellis2 import models as _models
from trellis2.modules import sparse as sp
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from terzi import render as BR
from common import bake_cond, first_png

VIEWS = ("front", "back", "right", "left")
N_IMG = 1 + len(VIEWS)

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
ap.add_argument("--data", default="two_branch_data", help="root of the prepared data (training/data/README.md)")
ap.add_argument("--init", default=None,
                help="start from this checkpoint instead of the stock pretrained init (accepts a ckpt of "
                     "this script: denoiser.* params + view_embed). Only used when not resuming.")
ap.add_argument("--smoke", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
KIND = {
    "slat": dict(config="configs/slat.json",
                 init=f"{a.data}/shape_body_aligned/pretrained_denoiser.pt",
                 wds=f"{a.data}/wds/shape_garment_aligned",
                 val_latents=f"{a.data}/shape_garment_aligned/shape_latent"),
    "ss": dict(config="configs/ss.json",
               init=f"{a.data}/ss_body_aligned/pretrained_ss_flow.pt",
               wds=f"{a.data}/wds/ss_aligned",
               val_latents=f"{a.data}/ss_garment_aligned/ss_latent"),
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

    def cond_tokens(self, rgb_tok, body_tok):
        """rgb_tok (B,L,C) at full resolution + body_tok (B,4,L,C) pooled ->
        (B, L + 4*L', C), each image tagged with its learned embedding."""
        e = self.view_embed.to(rgb_tok.dtype)
        body = pool_patches(body_tok, a.body_token_pool) + e[None, 1:, None, :]
        B, N, Lp, Cc = body.shape
        return torch.cat([rgb_tok + e[None, 0, None, :], body.reshape(B, N * Lp, Cc)], 1)

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
    src = a.init or KIND["init"]
    sd = torch.load(os.path.join(_WD, src) if not os.path.isabs(src) else src, map_location="cpu")
    if any(k.startswith("denoiser.") for k in sd):          # a checkpoint of this script
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
def cond_from(rgb, body_coords, body_feats):
    """rgb (B,3,512,512) in [0,1]; body_* lists of numpy -> DINO feats (B, N_IMG, L, C)."""
    B = rgb.shape[0]
    coords = torch.cat([torch.cat([torch.full((len(c), 1), i, dtype=torch.int32), torch.as_tensor(c).int()], 1)
                        for i, c in enumerate(body_coords)]).to(dev)
    feats = torch.cat([torch.as_tensor(f).float() for f in body_feats]).to(dev)
    meshes = body_dec(sp.SparseTensor(coords=coords, feats=feats))
    views = torch.stack([BR.render_views(m.vertices, m.faces, VIEWS) for m in meshes])   # B,4,3,S,S
    imgs = torch.cat([rgb[:, None], views], 1).reshape(B * N_IMG, 3, 512, 512)
    return dino(imgs).reshape(B, N_IMG, -1, dino_dim())


def dino_dim():
    return mcfg["args"]["cond_channels"]


# ---------------------------------------------------------------- data
def build_body_index():
    p = Path(_HERE) / "cache" / "body_index.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    idx = {}
    for tp in sorted(glob.glob(os.path.join(_WD, a.data, "wds/shape_body_aligned/*.tar"))):
        with tarfile.open(tp) as t:
            for m in t.getmembers():
                sha, ext = m.name.split(".", 1)
                if ext in ("coords.npy", "feats.npy"):
                    idx.setdefault(sha, {})[ext] = (tp, m.offset_data, m.size)
    if D.IS_MAIN:
        p.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(idx, open(str(p) + f".tmp{os.getpid()}", "wb")); os.replace(str(p) + f".tmp{os.getpid()}", p)
    return idx


if D.DDP:   # rank 0 builds the index once
    if D.IS_MAIN:
        BODY = build_body_index()
    dist.barrier()
    if not D.IS_MAIN:
        BODY = build_body_index()
else:
    BODY = build_body_index()
D.log(f"body index: {len(BODY)} shas")

_FH = {}
def _read(tp, off, size):
    if tp not in _FH:
        _FH[tp] = open(tp, "rb")
    f = _FH[tp]; f.seek(off); return f.read(size)


def _rgb(sample):
    img = Image.open(io.BytesIO(sample.get("cond.jpg") or sample["cond.png"])).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def decode(sample):
    sha = sample["__key__"]
    b = BODY.get(sha)
    if b is None or len(b) < 2:
        return None
    bc = np.load(io.BytesIO(_read(*b["coords.npy"]))); bf = np.load(io.BytesIO(_read(*b["feats.npy"])))
    if a.kind == "slat":
        c = torch.from_numpy(np.load(io.BytesIO(sample["coords.npy"]))).int()
        f = (torch.from_numpy(np.load(io.BytesIO(sample["feats.npy"]))).float() - MEAN) / STD
        return (c, f), _rgb(sample), bc, bf
    z = torch.from_numpy(np.load(io.BytesIO(sample["zgarment.npy"]))).float()
    return z, _rgb(sample), bc, bf


shards = sorted(glob.glob(os.path.join(_WD, KIND["wds"], "*.tar")))
D.log(f"shards: {len(shards)} ({KIND['wds']})")
ds = (wds.WebDataset(shards, resampled=True, shardshuffle=True, nodesplitter=wds.split_by_node,
                     handler=wds.ignore_and_continue)
      .shuffle(2000).map(decode, handler=wds.ignore_and_continue).select(lambda x: x is not None))
loader = iter(wds.WebLoader(ds, batch_size=None, num_workers=int(os.environ.get("GBC_WORKERS", "8")), prefetch_factor=4))


def collate(items):
    rgb = torch.stack([it[1] for it in items]).to(dev, non_blocking=True)
    bc = [it[2] for it in items]; bf = [it[3] for it in items]
    if a.kind == "slat":
        coords = torch.cat([torch.cat([torch.full((len(it[0][0]), 1), i, dtype=torch.int32), it[0][0]], 1)
                            for i, it in enumerate(items)]).to(dev)
        x0 = sp.SparseTensor(coords=coords, feats=torch.cat([it[0][1] for it in items]).to(dev))
    else:
        x0 = torch.stack([it[0] for it in items]).to(dev)
    return x0, rgb, bc, bf


def sample_t(n, g=None):
    if tsch["name"] == "logitNormal":
        return torch.sigmoid(torch.randn(n, generator=g) * tsch["args"]["std"] + tsch["args"]["mean"])
    return torch.rand(n, generator=g)


def fm_loss(x0, feats, drop, t, eps=None):
    """Flow-matching MSE exactly as the trainer (SLat: mean over all tokens of the batch)."""
    cond = model.cond_tokens(feats[:, 0], feats[:, 1:])
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
VAL = []
if D.IS_MAIN:
    val_shas = json.load(open(os.path.join(_WD, a.data, "wds/shape_garment_aligned/val_split.json")))["val"]
    bdir = Path(_WD) / a.data / "shape_body_aligned/shape_latent"
    gdir = Path(_WD) / KIND["val_latents"]
    rc = Path(_WD) / a.data / "shape_body_aligned/render_cond"
    for s in val_shas:
        if (bdir / f"{s}.npz").exists() and (gdir / f"{s}.npz").exists() and first_png(rc / s) is not None:
            VAL.append(s)
        if len(VAL) >= a.val_n:
            break
    D.log(f"val samples: {len(VAL)} (validation shas with garment + body latents)")


def val_item(s):
    rc = Path(_WD) / a.data / "shape_body_aligned/render_cond"
    _, b = bake_cond(first_png(rc / s), 512, "jpeg", 95)
    rgb = torch.from_numpy(np.array(Image.open(io.BytesIO(b)).convert("RGB"))).permute(2, 0, 1).float() / 255
    bd = np.load(Path(_WD) / a.data / "shape_body_aligned/shape_latent" / f"{s}.npz")
    g = np.load(Path(_WD) / KIND["val_latents"] / f"{s}.npz")
    if a.kind == "slat":
        x = (torch.tensor(g["coords"]).int(), (torch.tensor(g["feats"]).float() - MEAN) / STD)
    else:
        x = torch.tensor(g["z"]).float()
    return x, rgb, bd["coords"], bd["feats"]


@torch.no_grad()
def run_val():
    model.eval(); losses = []
    for i in range(0, len(VAL), micro):
        chunk = VAL[i:i + micro]
        x0, rgb, bc, bf = collate([val_item(s) for s in chunk])
        feats = cond_from(rgb, bc, bf)
        g = torch.Generator().manual_seed(zlib.crc32(",".join(chunk).encode()))
        t = sample_t(len(chunk), g).to(dev)
        eps_shape = x0.feats.shape if a.kind == "slat" else x0.shape
        eps = torch.randn(eps_shape, generator=g).to(dev)
        drop = torch.zeros(len(chunk), dtype=torch.bool, device=dev)     # conditional loss
        losses.append(float(fm_loss(x0, feats, drop, t, eps)) * len(chunk))
    model.train()
    return sum(losses) / max(1, len(VAL))


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
        t0 = time.time(); x0, rgb, bc, bf = collate([next(loader) for _ in range(micro)]); t_data += time.time() - t0
        t0 = time.time(); feats = cond_from(rgb, bc, bf); t_cond += time.time() - t0
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
        vl = run_val()
        is_best = vl < best_val - 1e-6
        if is_best:
            best_val = vl
        rec = {"step": step, "val_loss": vl, "best": is_best}
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
