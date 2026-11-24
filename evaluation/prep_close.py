"""CloSe-Di test split -> inputs for our --wild path. Per scan (metric = points / scale, SMPL-registered, y-up):
 one front view (camera 4 m in front of the SMPL body's facing direction, chest height, fov 30 deg, 1024 x 1024), unlit vertex
 colours (the scan's own albedo-like colours), white background; person alpha, garment mask (all garment labels), GT in the
 camera frame (OpenCV: x right, y down, z forward).  -> <out>/<id>/{input.jpg, mask_p1.png, gmask_p1_all.png, cam.json, gt.npz}
CloSe labels: 0 Hat 1 Body 2 Shirt 3 TShirt 4 Vest 5 Coat 6 Dress 7 Skirt 8 Pants 9 ShortPants 10 Shoes 11 Hoodies 12 Hair
13 Swimwear 14 Underwear 15 Scarf 16 Jumpsuits 17 Jacket.  Garment = every label except Hat, Body, Shoes, Hair.
  python evaluation/prep_close.py --close <CloSe-D root with CloSe-Di/ and test_ids.json> --out data/close_bench"""
import argparse, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr
ap = argparse.ArgumentParser(); ap.add_argument("--close", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
D = Path(a.close); OUT = Path(a.out); S = 1024; FOV = 30.0; DIST = 4.0
NOT_GARMENT = {0, 1, 10, 12}; ctx = dr.RasterizeCudaContext()
f = S / 2 / np.tan(np.radians(FOV) / 2); K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1.0]])
for sid in [x[:-4] for x in json.load(open(D / "test_ids.json"))]:
    if not (D / f"CloSe-Di/{sid}.npz").exists(): continue
    d = np.load(D / f"CloSe-Di/{sid}.npz"); od = OUT / sid
    if (od / "gt.npz").exists(): continue
    od.mkdir(parents=True, exist_ok=True)
    P = d["points"] / float(d["scale"]); F = d["faces"].astype(np.int32); C = d["colors"].astype(np.float32); L = d["labels"]
    go = Rot.from_rotvec(d["pose"][:3]).as_matrix(); fwd = go[:, 2].copy(); fwd[1] = 0; fwd /= np.linalg.norm(fwd)
    c = 0.5 * (P.min(0) + P.max(0)); cam = c + DIST * fwd
    z = -fwd; yd = np.array([0, -1.0, 0]); x = np.cross(yd, z); x /= np.linalg.norm(x); yd = np.cross(z, x); R = np.stack([x, yd, z])
    Pc = (P - cam) @ R.T                                                    # camera frame
    u = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]; v = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]; zn = (Pc[:, 2] - 1) / 10 * 2 - 1
    pos = torch.tensor(np.stack([2 * u / S - 1, 2 * v / S - 1, zn, np.ones_like(zn)], 1), dtype=torch.float32, device="cuda")[None]
    tri = torch.tensor(F, device="cuda"); rast, _ = dr.rasterize(ctx, pos.contiguous(), tri, resolution=[S, S])
    col, _ = dr.interpolate(torch.tensor(C, device="cuda")[None], rast, tri)
    a = (rast[..., 3:] > 0).float(); rgb = dr.antialias(torch.cat([col, a], -1), rast, pos, tri)[0].cpu().numpy()
    img = rgb[..., :3] * rgb[..., 3:] + (1 - rgb[..., 3:]); al = rgb[..., 3]
    tid = rast[0, ..., 3].long().cpu().numpy(); fl = L[F]; face_g = np.array([np.bincount(l, minlength=18).argmax() not in NOT_GARMENT for l in fl])
    gm = np.zeros((S, S), bool); gm[tid > 0] = face_g[tid[tid > 0] - 1]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(od / "input.jpg", quality=95)
    Image.fromarray((al > 0.5).astype(np.uint8) * 255).save(od / "mask_p1.png"); Image.fromarray(gm.astype(np.uint8) * 255).save(od / "gmask_p1_all.png")
    json.dump(dict(K=K.tolist(), R=R.tolist(), cam=cam.tolist()), open(od / "cam.json", "w"))
    np.savez_compressed(od / "gt.npz", V=Pc.astype(np.float32), F=F, L=L.astype(np.int8))
    print("prepared", sid, "garment px %.1f%%" % (100 * gm.mean()), flush=True)
