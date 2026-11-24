#!/usr/bin/env python3
"""Differentiable LAYER penalty between two garments (open surfaces), for finetune_masked.py.

A garment is an open sheet, so "inside" is only defined locally: for a query q we take
its nearest point x_c on the other garment G (cuBVH, detached) and G's face normal there,
oriented AWAY FROM THE BODY (outward). Then

    s(q) = (q - x_c) . n_out(x_c)          grad-live in q, like pen_loss.PenaltyBody

  q on an OUTER garment w.r.t. G (G is inner):  want s >= +clearance
  q on an INNER garment w.r.t. G (G is outer):  want s <= -clearance

A query only counts where G is actually there: |q - x_c| <= r_max and x_c not on a
boundary face of G (past G's hem/cuff the nearest point is the hem and the normal says
nothing about layering). Same Huber hinge as pen_loss.penetration_penalty.

All coordinates in metric crop px (common.to_metric), like pen_loss.
"""
import torch
import torch.nn.functional as F


def boundary_faces(faces, n_faces):
    """(F,) bool: faces with at least one edge used by exactly one face."""
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0).sort(1).values
    key = e[:, 0] * (int(f.max()) + 1) + e[:, 1]
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    on_bnd = (cnt[inv] == 1).view(3, -1).any(0)
    return on_bnd


class LayerSurface:
    """Frozen garment G -> local signed distance of queries along G's outward normal."""

    def __init__(self, v_metric, faces, body, r_max=8.0):
        """v_metric (V,3) px, faces (F,3), body: pen_loss.PenaltyBody of the same sample."""
        import cumesh
        v = v_metric.detach().float().contiguous(); f = faces.long().contiguous()
        self.v, self.f, self.r_max = v, f, r_max
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        n = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1, eps=1e-12)
        c = (v0 + v1 + v2) / 3.0
        with torch.no_grad():
            xb, _ = body.bb.project(c)                     # nearest body point per face
        n = torch.where(((c - xb) * n).sum(1, keepdim=True) < 0, -n, n)
        self.fn = n
        self.bnd = boundary_faces(f, len(f))
        self.bvh = cumesh.cuBVH(v, f.int())

    @torch.no_grad()
    def _nearest(self, q):
        d, fid, uvw = self.bvh.unsigned_distance(q.detach().float(), return_uvw=True)
        fid = fid.long()
        x_c = (self.v[self.f[fid]] * uvw[..., None]).sum(1)
        valid = (d <= self.r_max) & ~self.bnd[fid]
        return x_c, self.fn[fid], valid

    def signed(self, q):
        x_c, n, valid = self._nearest(q)
        return ((q - x_c) * n).sum(1), valid


class LocalOrder:
    """GT layer order of one garment pair as a FIELD, not one bit per pair (tucked / half-tucked tops, a jacket
    open at the front over a shirt...). pts: GT overlap points (metric px), f_i: smoothed share of GT votes saying
    garment i is the OUTER one near that point; maj_i: the pair's overall share (fallback away from GT overlap)."""

    def __init__(self, pts, f_i, maj_i, r_look=12.0, conf=0.3):
        self.p, self.f, self.maj, self.r, self.conf = pts.float(), f_i.float(), float(maj_i), r_look, conf

    @torch.no_grad()
    def want(self, q, is_i, chunk=8192):
        """(V,) +1 = q's garment must be OUTER here, -1 = inner, 0 = no constraint (tuck line / no information)"""
        d = torch.empty(len(q), device=q.device); idx = torch.empty(len(q), dtype=torch.long, device=q.device)
        for k in range(0, len(q), chunk):
            dd, ii = torch.cdist(q[k:k + chunk].detach().float(), self.p).min(1); d[k:k + chunk], idx[k:k + chunk] = dd, ii
        pi = torch.where(d <= self.r, self.f[idx], torch.full_like(d, self.maj))
        po = pi if is_i else 1.0 - pi
        return torch.where(po >= 1 - self.conf, 1.0, torch.where(po <= self.conf, -1.0, 0.0))


def layer_penalty(q, surf, outer, clearance=0.13, deep_clip=2.0, reduce_n=None):
    """q (V,3) grad-live vertices of the trained garment; surf: LayerSurface of the other
    garment; outer=True if q's garment is the OUTER one of the pair, or a (V,) tensor of
    +1 (outer here) / -1 (inner here) / 0 (unconstrained) from LocalOrder.want."""
    s, valid = surf.signed(q)
    if torch.is_tensor(outer):
        s_dir = s * outer; valid = valid & (outer != 0)
    else:
        s_dir = s if outer else -s                  # >= clearance is the correct side
    viol = F.relu(clearance - s_dir) * valid
    qv = torch.clamp(viol, max=deep_clip)
    pen = qv * qv + 2.0 * deep_clip * F.relu(viol - deep_clip)
    n = reduce_n if reduce_n is not None else max(int(q.shape[0]), 1)
    nv = int(valid.sum())
    with torch.no_grad():
        wrong = (s_dir < 0) & valid
    stats = {"overlap": nv / max(int(q.shape[0]), 1),
             "cross": float(wrong.sum()) / max(nv, 1),                       # of the overlapping vertices
             "cross_all": float(wrong.sum()) / max(int(q.shape[0]), 1)}      # of all vertices
    return pen.sum() / n, stats
