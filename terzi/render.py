"""Body condition renders (supplementary Sec. 6.1.4).

The body is rendered in the cube with an orthographic camera at 512 x 512 from four yaw angles about the cube's vertical axis:
front (0), back (180), right (+90) and left (-90). Flat per-face normals, two-sided towards the viewer, coloured as (n + 1) / 2
on a black background. The front view is pixel-aligned with the RGB crop; in the side views the horizontal axis is the
pelvis-relative depth in crop pixels (the depth gain is undone, so the body keeps its true proportions).
"""
import math

import nvdiffrast.torch as dr
import torch

from .crop import CROP as S, INV_SCALE_UV, INV_SCALE_Z

VIEWS = ("front", "back", "right", "left")
YAWS = {"front": 0.0, "back": 180.0, "right": 90.0, "left": -90.0}
_CTX = {}


def _ctx(device):
    if device not in _CTX:
        _CTX[device] = dr.RasterizeCudaContext(device=device)
    return _CTX[device]


def render_views(verts, faces, views=VIEWS):
    """verts (N, 3) cube coordinates on CUDA, faces (M, 3) int -> (len(views), 3, 512, 512) normal maps in [0, 1]."""
    dev = verts.device
    v = verts.float()
    p = torch.stack([v[:, 0] * INV_SCALE_UV, v[:, 1] * INV_SCALE_UV, v[:, 2] * INV_SCALE_Z], 1)
    f = faces.int().contiguous()
    fl = f.long()
    out = []
    for name in views:
        th = math.radians(YAWS[name])
        c, s = math.cos(th), math.sin(th)
        x = c * p[:, 0] + s * p[:, 2]
        z = -s * p[:, 0] + c * p[:, 2]
        y = p[:, 1]
        pos = torch.stack([x / (S / 2), -y / (S / 2), z / (S * 4), torch.ones_like(x)], 1)[None].contiguous()
        rast, _ = dr.rasterize(_ctx(dev), pos, f, resolution=[S, S])
        vr = torch.stack([x, y, z], 1)
        fn = torch.cross(vr[fl[:, 1]] - vr[fl[:, 0]], vr[fl[:, 2]] - vr[fl[:, 0]], dim=1)
        fn = torch.nn.functional.normalize(fn, dim=1)
        fn = torch.where(fn[:, 2:3] > 0, -fn, fn)
        n = torch.cat([torch.zeros(1, 3, device=dev), fn])[rast[0, ..., 3].long()]
        img = (n * torch.tensor([0.5, -0.5, -0.5], device=dev) + 0.5)[None]
        img = torch.flip(img * (rast[..., 3:4] > 0).float(), dims=[1])     # nvdiffrast rows are bottom-up
        out.append(img[0].permute(2, 0, 1))
    return torch.stack(out)
