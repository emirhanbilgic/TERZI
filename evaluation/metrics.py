"""Evaluation protocol of the paper (supplementary Sec. 9): alignment, the two protocols, CD / NC / F@1, penetration, holes.

  alignment       the complete prediction (garments + its body) is aligned to the complete scan with ICP with uniform scale;
                  4D-DRESS: one start from the best of the 4 PCA sign combinations (align_single), CloSe / BEDLAM2.0: ICP from
                  each of the 4 PCA sign starts, best fit kept (align_multi). The garments inherit the transformation.
  segmentation    our garment mesh as generated (hidden parts included) against the scan's garment surface
  GT label transfer   the visible shell of the aligned prediction (orthographic face-id renders from 36 x 7 directions),
                  each face labelled by the nearest scan triangle; the faces labelled as clothing are scored
  CD / NC / F@1   100k points per surface; CD = mean of the two directed nearest-neighbour distances, NC = mean |cos| of the
                  normals of nearest neighbours (both directions), F@1 = F-score at 1 cm
  penetration     % of the garment surface (100k points) inside the conditioning body
  holes           boundary loops of the raw garment mesh with a perimeter of at least 10 cm
"""
import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

N = 100_000
TAU_M = 0.01


def sith_metrics(pred, gt, seed):
    p, ip = trimesh.sample.sample_surface(pred, N, seed=seed)
    g, ig = trimesh.sample.sample_surface(gt, N, seed=seed + 1)
    pn, gn = pred.face_normals[ip], gt.face_normals[ig]
    d1, k1 = cKDTree(g).query(p, workers=-1)
    d2, k2 = cKDTree(p).query(g, workers=-1)
    nc = 0.5 * (np.abs((pn * gn[k1]).sum(1)).mean() + np.abs((gn * pn[k2]).sum(1)).mean())
    prec, rec = (d1 < TAU_M).mean(), (d2 < TAU_M).mean()
    return {"p2s_mm": 1000 * d1.mean(), "s2p_mm": 1000 * d2.mean(), "cd_mm": 500 * (d1.mean() + d2.mean()), "nc": nc,
            "fscore": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)}


def pca_axes(pts):
    c = pts - pts.mean(0)
    w, v = np.linalg.eigh(c.T @ c / max(len(c) - 1, 1))
    o = np.argsort(w)[::-1]
    return v[:, o].T, w[o]


def umeyama(p, g, allow_scale=True):
    pm, gm = p.mean(0), g.mean(0)
    p0, g0 = p - pm, g - gm
    U, S, Vt = np.linalg.svd(p0.T @ g0)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / max((p0 ** 2).sum(), 1e-12) if allow_scale else 1.0
    return float(s), R, gm - s * (R @ pm)


def _pca_start(pred_mesh, gt_mesh, n_samples=10_000, seed=7):
    """best of the 4 PCA sign starts by the symmetric surface distance (no ICP iterations)."""
    np.random.seed(seed)
    P = trimesh.sample.sample_surface(pred_mesh, n_samples)[0]
    G = trimesh.sample.sample_surface(gt_mesh, n_samples)[0]
    pa, pv = pca_axes(P)
    ga, gv = pca_axes(G)
    pm, gm = P.mean(0), G.mean(0)
    s = float(np.sqrt(max(gv[0], 1e-12) / max(pv[0], 1e-12)))
    tg = cKDTree(G)
    best = (np.inf, None)
    for sx in (+1, -1):
        for sy in (+1, -1):
            ax = pa * np.array([sx, sy, 1.0])[:, None]
            R = ga.T @ ax
            if np.linalg.det(R) < 0:
                ax[2] *= -1
                R = ga.T @ ax
            t = gm - s * (R @ pm)
            cur = (s * (R @ P.T)).T + t
            c = float(tg.query(cur)[0].mean() + cKDTree(cur).query(G)[0].mean()) / 2
            if c < best[0]:
                best = (c, (s, R, t))
    return best[1]


def align_single(full_pred, scan):
    """4D-DRESS: start from the best PCA sign combination, then ICP with scale (trimesh defaults). Returns a 4x4 transform."""
    s0, R0, t0 = _pca_start(full_pred, scan)
    T0 = np.eye(4)
    T0[:3, :3] = s0 * R0
    T0[:3, 3] = t0
    M, _, _ = trimesh.registration.icp(trimesh.transform_points(np.asarray(full_pred.vertices), T0), np.asarray(scan.vertices))
    return M @ T0


def align_multi(full_pred, scan, n=20_000, seed=3):
    """CloSe / BEDLAM2.0: ICP with scale from each of the 4 PCA sign starts, lowest symmetric surface distance kept."""
    P = trimesh.sample.sample_surface(full_pred, n, seed=seed)[0]
    G = trimesh.sample.sample_surface(scan, n, seed=seed + 1)[0]
    tg = cKDTree(G)
    pa, pv = pca_axes(P)
    ga, gv = pca_axes(G)
    pm, gm = P.mean(0), G.mean(0)
    s0 = float(np.sqrt(max(gv[0], 1e-12) / max(pv[0], 1e-12)))
    best = None
    for sx in (+1, -1):
        for sy in (+1, -1):
            ax = pa * np.array([sx, sy, 1.0])[:, None]
            R = ga.T @ ax
            if np.linalg.det(R) < 0:
                ax[2] *= -1
                R = ga.T @ ax
            T0 = np.eye(4)
            T0[:3, :3] = s0 * R
            T0[:3, 3] = gm - s0 * (R @ pm)
            M, _, _ = trimesh.registration.icp(trimesh.transform_points(np.asarray(full_pred.vertices), T0), np.asarray(scan.vertices))
            T = M @ T0
            Q = trimesh.transform_points(P, T)
            c = 500 * (tg.query(Q)[0].mean() + cKDTree(Q).query(G)[0].mean())
            if best is None or c < best[0]:
                best = (c, T)
    return best[1]


_CTX = {}


def _ctx(dev):
    if dev not in _CTX:
        import nvdiffrast.torch as dr
        _CTX[dev] = dr.RasterizeCudaContext(device=dev)
    return _CTX[dev]


def visible_part(mesh, dev="cuda", res=1536, n_az=36, elevs=(-60, -30, 0, 30, 60, 85, -85)):
    """faces of `mesh` seen from outside: union of the faces winning a pixel in orthographic face-id renders from
    n_az x len(elevs) directions."""
    import nvdiffrast.torch as dr
    if mesh is None or not len(mesh.faces):
        return mesh
    v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=dev)
    f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=dev)
    v = v - v.mean(0)
    v = v / (v.norm(dim=1).max() * 1.05)
    seen = torch.zeros(len(f), dtype=torch.bool, device=dev)
    for el in elevs:
        for az in np.linspace(0, 360, n_az, endpoint=False):
            a, e = np.radians(az), np.radians(el)
            Ry = torch.tensor([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=torch.float32, device=dev)
            Rx = torch.tensor([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]], dtype=torch.float32, device=dev)
            p = v @ (Rx @ Ry).T
            clip = torch.cat([p[:, :2], -p[:, 2:3], torch.ones_like(p[:, :1])], 1)[None].contiguous()
            rast, _ = dr.rasterize(_ctx(dev), clip, f, resolution=[res, res])
            ids = rast[0, ..., 3].long()
            seen[ids[ids > 0] - 1] = True
    keep = seen.cpu().numpy()
    if not keep.any():
        return None
    used = np.unique(np.asarray(mesh.faces)[keep].ravel())
    remap = np.full(len(mesh.vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(np.asarray(mesh.vertices)[used], remap[np.asarray(mesh.faces)[keep]], process=False)


def label_transfer_cloth(full_aligned, scan_v, scan_f, scan_labels, cloth_labels, dev="cuda"):
    """GT label transfer: the visible shell of the aligned prediction, each face labelled by the nearest scan triangle
    (label of its closest corner); returns the faces labelled as clothing (or None)."""
    import cumesh
    outer = visible_part(full_aligned, dev)
    if outer is None or not len(outer.faces):
        return None
    bvh = cumesh.cuBVH(torch.as_tensor(np.asarray(scan_v, np.float32), device=dev), torch.as_tensor(np.asarray(scan_f, np.int32), device=dev))
    _, fid, uvw = bvh.unsigned_distance(torch.as_tensor(np.asarray(outer.triangles_center, np.float32), device=dev), return_uvw=True)
    lab = np.asarray(scan_labels)[np.asarray(scan_f)[fid.long().cpu().numpy(), uvw.argmax(1).cpu().numpy()]]
    is_cloth = np.isin(lab, list(cloth_labels))
    if not is_cloth.any():
        return None
    return outer.submesh([np.nonzero(is_cloth)[0]], append=True)


class BodySDF:
    """signed distance to a body mesh (cumesh BVH, ray stabbing; negative inside)."""

    def __init__(self, body, dev="cuda"):
        import cumesh
        b = body.copy()
        try:
            b.fix_normals()
        except Exception:
            pass
        try:
            if b.volume < 0:
                b.invert()
        except Exception:
            pass
        self.bvh = cumesh.cuBVH(torch.tensor(np.asarray(b.vertices), dtype=torch.float32, device=dev),
                                torch.tensor(np.asarray(b.faces), dtype=torch.int32, device=dev))
        self.dev, self.flip = dev, None

    def __call__(self, q):
        s, _, _ = self.bvh.signed_distance(torch.as_tensor(np.asarray(q), dtype=torch.float32, device=self.dev), mode="raystab")
        if self.flip is None:
            self.flip = bool((s < 0).float().mean() > 0.5)
        return (-s if self.flip else s).cpu().numpy()


def penetration(garment, body, seed, dev="cuda"):
    """share of the garment surface inside the body (100k surface points)."""
    p, _ = trimesh.sample.sample_surface(garment, N, seed=seed)
    return float((BodySDF(body, dev)(p) < 0).mean())


def holes(mesh, min_perimeter=0.10):
    """boundary loops of `mesh` with a perimeter of at least `min_perimeter` metres (natural openings included)."""
    F = np.asarray(mesh.faces)
    V = np.asarray(mesh.vertices)
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    bd = u[c == 1]
    if not len(bd):
        return 0
    vid, iv = np.unique(bd.ravel(), return_inverse=True)
    iv = iv.reshape(-1, 2)
    n = len(vid)
    k, lab = connected_components(coo_matrix((np.ones(len(iv)), (iv[:, 0], iv[:, 1])), shape=(n, n)), directed=False)
    per = np.bincount(lab[iv[:, 0]], np.linalg.norm(V[bd[:, 0]] - V[bd[:, 1]], axis=1), minlength=k)
    return int((per >= min_perimeter).sum())
