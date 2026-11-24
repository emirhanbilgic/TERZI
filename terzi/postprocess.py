"""Hole filling of the generated garment (the shape decoder of TRELLIS.2 can leave small holes and detached seams).

  fast    every hole with a perimeter below 10 cm filled (duplicate faces removed first): the post-processing of all reported
          numbers (supplementary Sec. 9.6)
  patch   fast, then body-guided patches: from every body vertex, rays along the body normal are cast against the garment;
          body regions that the garment neither covers (outward hit within 25 cm) nor is pierced by (inward hit within 3 cm) and that
          do not touch the head, neck, hands or feet are holes. Each hole smaller than 2% of the body surface is closed with a patch:
          the body faces of the hole and a ring around it, pushed outward to the cloth (offset interpolated harmonically from the
          surrounding covered vertices, at most 3 cm and at most half the free space in front of the body). Larger bare regions
          are kept (real skin). Needs an SMPL-X body (the joint each vertex belongs to).
"""
import numpy as np
import scipy.sparse as sp
import torch
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

N_SMPLX = 10475
EXTREMITIES = [7, 8, 10, 11, 12, 15, 20, 21, 22, 23, 24] + list(range(25, 55))     # SMPL-X joints of feet, neck, head, hands


def fill_holes(mesh, max_perimeter=0.10):
    """fast: duplicate faces removed, every hole with a perimeter below `max_perimeter` metres filled."""
    import cumesh
    cm = cumesh.CuMesh()
    cm.init(torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda"),
            torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32, device="cuda"))
    cm.remove_duplicate_faces()
    cm.fill_holes(max_hole_perimeter=max_perimeter)
    v, f = cm.read()
    return trimesh.Trimesh(v.cpu().numpy().astype(np.float64), f.cpu().numpy().astype(np.int64), process=False)


def _vnormals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    outward = (V[F[:, 0]] * np.cross(V[F[:, 1]], V[F[:, 2]])).sum() > 0
    return n * (1.0 if outward else -1.0), outward


def _adjacency(F, n):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return ((A + A.T) > 0).astype(np.float64).tocsr()


def _bvh(V, F):
    import cumesh
    return cumesh.cuBVH(torch.tensor(V, dtype=torch.float32, device="cuda"), torch.tensor(F, dtype=torch.int32, device="cuda"))


def _classify(bv, bf, gv, gf, extremity, reach, inner):
    """per body vertex: 0 covered, 1 skin, 2 penetration, 3 hole; outward cloth distance; normals; adjacency."""
    n, _ = _vnormals(bv, bf)
    bvh = _bvh(gv, gf)
    o = torch.tensor(bv, dtype=torch.float32, device="cuda")
    d = torch.tensor(n, dtype=torch.float32, device="cuda")
    _, fo, to = bvh.ray_trace(o + 1e-3 * d, d)
    _, fi, ti = bvh.ray_trace(o - 1e-3 * d, -d)
    to, ti, fo, fi = to.cpu().numpy(), ti.cpu().numpy(), fo.cpu().numpy(), fi.cpu().numpy()
    out_hit = (fo >= 0) & np.isfinite(to) & (to < reach)
    in_hit = (fi >= 0) & np.isfinite(ti) & (ti < inner)
    lab = np.ones(len(bv), np.int8)
    lab[out_hit] = 0
    lab[~out_hit & in_hit] = 2
    exp = lab == 1
    A = _adjacency(bf, len(bv))
    nc, cc = connected_components(A[exp][:, exp], directed=False)
    idx = np.nonzero(exp)[0]
    ext = np.zeros(nc, bool)
    np.logical_or.at(ext, cc, extremity[idx])
    lab[idx[~ext[cc]]] = 3
    return lab, np.where(out_hit, to, np.nan), n, A


def _big_holes_to_skin(bv, bf, lab, A, max_area):
    hole = lab == 3
    if not hole.any() or max_area >= 1.0:
        return lab
    fa = 0.5 * np.linalg.norm(np.cross(bv[bf[:, 1]] - bv[bf[:, 0]], bv[bf[:, 2]] - bv[bf[:, 0]]), axis=1)
    va = np.zeros(len(bv))
    for k in range(3):
        np.add.at(va, bf[:, k], fa / 3)
    idx = np.nonzero(hole)[0]
    nc, cc = connected_components(A[hole][:, hole], directed=False)
    area = np.bincount(cc, weights=va[idx], minlength=nc) / va.sum()
    lab = lab.copy()
    lab[idx[(area >= max_area)[cc]]] = 1
    return lab


def _patch(bv, bf, lab, t_out, n, A, ring, max_off):
    hole = lab == 3
    if not hole.any():
        return None
    reg = hole.copy()
    for _ in range(ring):
        reg = reg | (A @ reg.astype(float) > 0)
    fsel = reg[bf].all(1)
    Fp = bf[fsel]
    used = np.unique(Fp)
    mp = -np.ones(len(bv), np.int64)
    mp[used] = np.arange(len(used))
    rim = used[~hole[used] & np.isfinite(t_out[used]) & (t_out[used] <= max_off)]   # covered, cloth close: known offset
    if len(rim) == 0:
        return None
    Ap = A[used][:, used]
    L = sp.diags(np.asarray(Ap.sum(1)).ravel()) - Ap
    fixed = np.isin(used, rim)
    free = ~fixed
    off = np.zeros(len(used))
    off[fixed] = t_out[used[fixed]]
    if free.any():
        rhs = -L[free][:, fixed] @ off[fixed]
        off[free] = spsolve((L[free][:, free] + sp.eye(int(free.sum())) * 1e-9).tocsc(), rhs)
    off = np.clip(off - 0.002, 0.003, max_off)
    # free space in front of each patch vertex (concave places face another limb): at most half of it
    d_ = torch.tensor(n[used], dtype=torch.float32, device="cuda")
    o_ = torch.tensor(bv[used], dtype=torch.float32, device="cuda") + 2e-3 * d_
    _, fb, tb = _bvh(bv, bf).ray_trace(o_, d_)
    tb, fb = tb.cpu().numpy(), fb.cpu().numpy()
    tb = np.where((fb >= 0) & np.isfinite(tb), tb + 2e-3, np.inf)
    off = np.minimum(off, 0.5 * tb - 5e-4)
    Vp, Fq = bv[used] + n[used] * np.maximum(off, 0.0)[:, None], mp[Fp]
    Fq = Fq[(off[Fq] > 1e-3).all(1)]
    if not len(Fq):
        return None
    sd = _signed(bv, bf, Vp)                                                         # drop patch faces inside another body part
    Fq = Fq[(sd[Fq] > 0.001).all(1)]
    if not len(Fq):
        return None
    u, inv = np.unique(Fq, return_inverse=True)
    return Vp[u], inv.reshape(-1, 3)


def _signed(bv, bf, q):
    s, _, _ = _bvh(bv, bf).signed_distance(torch.tensor(q, dtype=torch.float32, device="cuda"), mode="raystab")
    s = s.cpu().numpy()
    return -s if (s < 0).mean() > 0.5 else s


def patch_holes(garment, body, joint_of_vertex, reach=0.25, inner=0.03, ring=2, max_off=0.03, max_area=0.02):
    """body-guided patches (see the module docstring). garment, body: trimesh meshes in the same frame (metres);
    joint_of_vertex: (10475,) SMPL-X joint index of each body vertex (argmax of the skinning weights)."""
    bv, bf = np.asarray(body.vertices, np.float64), np.asarray(body.faces, np.int64)
    gv, gf = np.asarray(garment.vertices, np.float64), np.asarray(garment.faces, np.int64)
    lab, t_out, n, A = _classify(bv, bf, gv, gf, np.isin(joint_of_vertex, EXTREMITIES), reach, inner)
    lab = _big_holes_to_skin(bv, bf, lab, A, max_area)
    pr = _patch(bv, bf, lab, t_out, n, A, ring, max_off)
    if pr is None:
        return garment
    Vp, Fp = pr
    if not _vnormals(bv, bf)[1]:
        Fp = Fp[:, ::-1]
    return trimesh.Trimesh(np.concatenate([gv, Vp]), np.concatenate([gf, Fp + len(gv)]), process=False)
