#!/usr/bin/env python3
"""Differentiable body-penetration penalty on the SLat decoder's output.

WHY THIS IS DIFFERENTIABLE
----------------------------------------------------
trellis2/models/sc_vaes/fdg_vae.py :: FlexiDualGridVaeDecoder.forward (eval)

    vertices     = (1 + 2*margin) * sigmoid(h.feats[..., 0:3]) - margin   # grad
    intersected  = h.feats[..., 3:6] > 0                                  # HARD
    quad_lerp    = softplus(h.feats[..., 6:7])                            # grad
    mesh         = flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp, ...)

and inside o_voxel/convert/flexible_dual_grid.py the returned positions are

    mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]

i.e. an affine function of the sigmoid output.  Everything non-differentiable
(`intersected`, the hashmap neighbour lookup, the quad-split `torch.where`)
only ever produces INDICES: it selects which dual vertices become mesh
vertices and how they are wired into triangles.  The positions themselves are
a clean autograd path back to the latent.  So we do NOT need a surrogate: the
penalty is built on exactly the vertices the evaluator measures.

Frames
------
  cube   [-0.5, 0.5]^3, image-aligned (crop_space.py)
  metric crop pixels, via common.to_metric; 1 px ~ 4.1 mm on
         4D-DRESS / BEDLAM crops (512^3 decoder cell == 1 px).
All distances below are in crop pixels, as in the evaluation.
"""
import torch
import torch.nn.functional as F


def fast_bvh(v_metric, faces, flip=None):
    """cuBVH straight from (vertices, faces) — no trimesh.

    common.BodyBVH routes through trimesh and calls fix_normals(), which
    costs ~1 s on a 20908-face SMPL-X mesh. At 829 samples that was amortised by
    the cache; at 20k it is the whole step time. Every body here is the SAME
    SMPL-X topology with the same winding, so the orientation test only has to be
    done once (pass the answer back in as `flip`); the area-weighted volume test
    is done analytically on the GPU instead of by trimesh.
    """
    import cumesh
    v = v_metric.float().contiguous()
    f = faces.long().contiguous()
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    if flip is None:                       # signed volume via the divergence theorem
        flip = bool((((v0 + v1 + v2) / 3.0) * fn).sum() / 6.0 < 0)
    if flip:
        f = f[:, [0, 2, 1]].contiguous()
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    b = _FastBVH()
    b.bv, b.bf = v, f
    b.fn = torch.nn.functional.normalize(fn, dim=1, eps=1e-12)
    b.bvh = cumesh.cuBVH(v, f.int())
    b.flip = None
    return b, flip


class _FastBVH:
    """Same surface as common.BodyBVH (signed / project / bv / bf / fn)."""

    def signed(self, q):
        s, _, _ = self.bvh.signed_distance(q.float(), mode="raystab")
        if self.flip is None:
            self.flip = bool((s < 0).float().mean() > 0.5)
        return -s if self.flip else s

    def project(self, q):
        _, fid, uvw = self.bvh.unsigned_distance(q.float(), return_uvw=True)
        tri = self.bv[self.bf[fid]]
        return (tri * uvw[..., None]).sum(dim=1), self.fn[fid]


class PenaltyBody:
    """Frozen conditioning body -> differentiable signed distance for queries.

    The nearest-point correspondence comes from the cuBVH (detached, an
    argmin); the projection onto the outward direction keeps the query in the
    autograd graph.  d/dq  signed(q) = outward(q), the correct push direction.
    """

    def __init__(self, bvh):
        self.bb = bvh            # common.BodyBVH, already in metric px

    limb_v = None      # optional (n_body_verts,) 0/1 mask, set by the trainer

    def limb_weight(self, q, extra):
        """1 + extra on garment vertices whose nearest body FACE is on a limb."""
        with torch.no_grad():
            _, fid, _ = self.bb.bvh.unsigned_distance(q.detach().float(), return_uvw=True)
            m = self.limb_v[self.bb.bf[fid.long()]].amax(dim=1)
        return 1.0 + extra * m

    def signed(self, q):
        with torch.no_grad():
            s0 = self.bb.signed(q.detach())
            x_c, n = self.bb.project(q.detach())
            dvec = q.detach() - x_c
            d = dvec.norm(dim=1, keepdim=True)
            sgn = torch.where(s0 >= 0, torch.ones_like(s0), -torch.ones_like(s0))[:, None]
            out = torch.where(d > 1e-4, dvec / d.clamp_min(1e-8) * sgn, n)
        return ((q - x_c) * out).sum(dim=1), s0


def penetration_penalty(verts, body, clearance=0.4, deep_clip=3.0,
                        region_w=None, reduce_n=None):
    """relu(clearance - s)^2, robustly clipped, averaged over `reduce_n`.

    verts      (V,3) metric px, grad-live
    clearance  target standoff in px (0.4 px ~ 1.6 mm)
    deep_clip  violations deeper than this are linearised (a 5 cm outlier from
               a bad body estimate must not own the gradient)
    region_w   optional (V,) weight (the diagnosis puts ~55% of penetration on
               21% of vertices, all on the limbs)
    """
    s, s_hard = body.signed(verts)
    viol = F.relu(clearance - s)
    # Huber-style: quadratic up to deep_clip, linear beyond
    q = torch.clamp(viol, max=deep_clip)
    pen = q * q + 2.0 * deep_clip * F.relu(viol - deep_clip)
    if region_w is not None:
        pen = pen * region_w
    n = reduce_n if reduce_n is not None else max(int(verts.shape[0]), 1)
    stats = {
        "frac_in": float((s_hard < 0).float().mean()),
        "frac_deep": float((s_hard < -0.5).float().mean()),
        "frac_band": float((s_hard < clearance).float().mean()),
        "mean_depth_px": float((-s_hard[s_hard < 0]).mean()) if bool((s_hard < 0).any()) else 0.0,
        "so_median_px": float(s_hard.median()),
    }
    return pen.sum() / n, stats


def surface_vertex_mask(faces, n_verts, device):
    """Which dual vertices the extraction actually wired into the mesh.

    The decoder emits one dual vertex per active fine voxel, but only those
    referenced by a quad are on the surface; the rest are free-floating and
    their positions carry no geometric meaning, so penalising them would be
    pure gradient noise.
    """
    m = torch.zeros(n_verts, dtype=torch.bool, device=device)
    m[faces.reshape(-1).long()] = True
    return m
