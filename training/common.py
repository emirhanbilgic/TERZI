"""Small helpers shared by the training, fine-tuning and distillation scripts."""
import io
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

import crop_space as _cs

ROOT = Path(__file__).resolve().parents[1]
INV_SCALE_UV = _cs.INV_SCALE_UV
INV_SCALE_Z = _cs.INV_SCALE_Z
SMPLX_PATH = os.environ.get("SMPLX_PATH", str(ROOT / "data" / "SMPLX_NEUTRAL.npz"))


def build_pipeline():
    """The stock TRELLIS.2 image-to-3D pipeline: frozen decoders/encoder, DINOv3, samplers and latent normalisation."""
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = False
    pipeline.cuda()
    pipeline.models["sparse_structure_decoder"].to(pipeline.device)
    pipeline.image_cond_model.to(pipeline.device)
    return pipeline


def to_metric(verts, y_up=True):
    """cube [-1/2, 1/2]^3 (anisotropic, y down) -> crop pixels with the body's true proportions (y up by default).
    A per-axis constant scale, so it is differentiable."""
    s = torch.tensor([INV_SCALE_UV, -INV_SCALE_UV if y_up else INV_SCALE_UV, INV_SCALE_Z], device=verts.device, dtype=verts.dtype)
    return verts * s


def _tm(v_metric, faces):
    return trimesh.Trimesh(np.ascontiguousarray(v_metric.detach().cpu().numpy(), np.float64),
                           np.ascontiguousarray(faces.cpu().numpy(), np.int64), process=False)


class BodyBVH:
    """Triangle-exact body queries with the cumesh BVH: signed distance (ray stabbing, robust on non-watertight bodies) and the
    closest surface point with its face normal."""

    def __init__(self, body_tm, device):
        import cumesh
        b = body_tm.copy()
        try:
            b.fix_normals()
        except Exception:
            pass
        try:
            if b.volume < 0:
                b.invert()
        except Exception:
            pass
        self.bv = torch.tensor(np.asarray(b.vertices), dtype=torch.float32, device=device)
        self.bf = torch.tensor(np.asarray(b.faces), dtype=torch.int64, device=device)
        self.fn = torch.tensor(np.asarray(b.face_normals), dtype=torch.float32, device=device)
        self.bvh = cumesh.cuBVH(self.bv, self.bf.int())
        self.flip = None            # most garment vertices are outside the body: fix the sign convention on the first query

    def signed(self, q):
        s, _, _ = self.bvh.signed_distance(q.float(), mode="raystab")
        if self.flip is None:
            self.flip = bool((s < 0).float().mean() > 0.5)
        return -s if self.flip else s

    def project(self, q):
        _, fid, uvw = self.bvh.unsigned_distance(q.float(), return_uvw=True)
        tri = self.bv[self.bf[fid]]
        return (tri * uvw[..., None]).sum(dim=1), self.fn[fid]


def mesh_openness(verts_np, faces_np):
    """number of boundary edges (edges used by exactly one face)."""
    f = np.asarray(faces_np)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum())


def first_png(d: Path):
    for p in sorted(Path(d).glob("*.png")):
        return p
    return None


def bake_cond(png_path, image_size=512, fmt="jpeg", quality=95):
    """the RGB condition as the trainer consumes it: resized, alpha premultiplied, JPEG q95. Returns (name, bytes)."""
    img = Image.open(png_path).resize((image_size, image_size), Image.Resampling.LANCZOS)
    a = np.asarray(img.getchannel(3), np.float32) / 255.0
    rgb = (np.asarray(img.convert("RGB"), np.float32) * a[..., None]).round().clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    if fmt == "jpeg":
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return "cond.jpg", buf.getvalue()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "cond.png", buf.getvalue()


def smplx_layer(device="cuda"):
    """neutral SMPL-X layer (10 betas, flat hands), with TF32 off as the bodies were generated."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import smplx
    return smplx.create(SMPLX_PATH, model_type="smplx", gender="neutral", use_pca=False, flat_hand_mean=True,
                        num_betas=10, num_expression_coeffs=10).to(device).eval()
