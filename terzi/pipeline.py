"""TERZI inference: body-conditioned garment generation (paper Secs. 3.1 and 3.4).

    pipe = TerziPipeline.from_pretrained("terzi_masked")             # or terzi, terzi_student, terzi_masked_student
    out = pipe(image, person_mask, K, body, garment_mask=mask)      # numpy arrays + terzi.Body
    out.garment.export("garment.glb")                                # camera frame, metres

Condition: the RGB crop, four orthographic normal renders of the body and, for the masked models, the garment mask, encoded by
a frozen DINOv3 into one cross-attention context. Stage 1 (SS DiT) generates the sparse structure, i.e. the cells the garment
occupies; Stage 2 (SLat DiT) generates the structured latent on them; the frozen TRELLIS.2 decoder turns it into the garment
mesh, which is mapped back to the camera frame, and its holes are filled (terzi/postprocess.py: "fast" = holes with a perimeter
below 10 cm, the setting of all reported numbers; "patch", the default = fast + body-guided patches over enclosed bare regions).
"""
import io
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from . import crop as C
from .body import Body
from .postprocess import fill_holes, patch_holes
from .render import render_views

REPO = "emirhanbilgic/TERZI"
MODELS = {   # folder in the Hugging Face repo -> (garment mask input, one-step student)
    "terzi": (False, False), "terzi_masked": (True, False),
    "terzi_student": (False, True), "terzi_masked_student": (True, True)}
TRELLIS2 = "microsoft/TRELLIS.2-4B"
SS_DECODER = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
SHAPE_DECODER = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
DINOV3 = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DIT = dict(model_channels=1536, cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334, pe_mode="rope", share_mod=True,
           initialization="scaled", qk_rms_norm=True, qk_rms_norm_cross=True, use_checkpoint=False, dtype="bfloat16")
SS_ARGS = dict(resolution=16, in_channels=8, out_channels=8, **DIT)            # Stage 1: 8 x 16^3 structure latent
SLAT_ARGS = dict(resolution=32, in_channels=32, out_channels=32, **DIT)        # Stage 2: 32 channels per occupied cell of 32^3
BODY_TOKEN_POOL = 2                                                             # body views: 2 x 2 pooled patch tokens


@dataclass
class TerziOutput:
    garment: trimesh.Trimesh                 # camera frame, metres, holes filled as requested
    body: trimesh.Trimesh                    # the conditioning body, camera frame, metres
    garment_cube: trimesh.Trimesh            # raw decoder output in the cube [-1/2, 1/2]^3
    cond: Image.Image                        # the 512 x 512 RGBA crop
    body_views: np.ndarray                   # (4, 512, 512, 3) uint8 normal renders
    mask: np.ndarray = None                  # (512, 512) bool garment mask crop (masked models)
    box: dict = field(default_factory=dict)  # crop box in image pixels {cx, cy, size}


def _seed(key, tag):
    v = zlib.crc32(f"{key}:{tag}".encode()) & 0xFFFFFFFF
    torch.manual_seed(v)
    torch.cuda.manual_seed_all(v)


def _bake(rgba):
    """premultiplied alpha + JPEG q95: the RGB condition exactly as in training."""
    a = np.asarray(rgba.getchannel(3), np.float32) / 255.0
    rgb = (np.asarray(rgba.convert("RGB"), np.float32) * a[..., None]).round().clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return torch.from_numpy(np.array(Image.open(buf).convert("RGB"))).permute(2, 0, 1).float() / 255


def _pool_patches(tok, f):
    """2 x 2 average pooling of the patch tokens (the first 5 tokens are the class and register tokens)."""
    n, l, c = tok.shape
    g = int(round((l - 5) ** 0.5))
    p = F.avg_pool2d(tok[:, 5:].reshape(n, g, g, c).permute(0, 3, 1, 2), f).permute(0, 2, 3, 1).reshape(n, -1, c)
    return torch.cat([tok[:, :5], p], 1)


def _cond_tokens(tok, view_embed):
    """[RGB tokens, pooled tokens of the 4 body views, (mask tokens)], each plus its learned view embedding."""
    e = view_embed.to(tok.dtype)
    body = _pool_patches(tok[1:5], BODY_TOKEN_POOL) + e[1:5, None, :]
    parts = [tok[0] + e[0], body.reshape(-1, body.shape[-1])]
    if len(view_embed) == 6:
        parts.append(tok[5] + e[5])
    return torch.cat(parts, 0)[None]


def _state_dict(path):
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


class TerziPipeline:
    def __init__(self, ss, slat, ve_ss, ve_slat, masked, student, device="cuda", dino_bf16=True):
        from trellis2 import models
        from trellis2.modules import image_feature_extractor
        from trellis2.pipelines import samplers
        self.device, self.masked, self.student = device, masked, student
        self.ss, self.slat, self.ve_ss, self.ve_slat = ss, slat, ve_ss, ve_slat
        self.ss_decoder = models.from_pretrained(SS_DECODER).eval().to(device)
        self.decoder = models.from_pretrained(SHAPE_DECODER).eval().to(device)
        self.decoder.set_resolution(512)
        self.dino = image_feature_extractor.DinoV3FeatureExtractor(DINOV3)
        self.dino.to(device)
        if dino_bf16:
            self.dino.model.to(torch.bfloat16)
        cfg = json.load(open(self._trellis2_config()))["args"]
        self.ss_sampler = samplers.FlowEulerGuidanceIntervalSampler(**cfg["sparse_structure_sampler"]["args"])
        self.slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(**cfg["shape_slat_sampler"]["args"])
        self.ss_params = dict(cfg["sparse_structure_sampler"]["params"])        # 12 steps, CFG 7.5, guidance interval [0.6, 1]
        self.slat_params = dict(cfg["shape_slat_sampler"]["params"])
        if student:                                                             # one step per stage, guidance baked in
            for p in (self.ss_params, self.slat_params):
                p["steps"], p["guidance_strength"] = 1, 1.0
        norm = cfg["shape_slat_normalization"]
        self.mean = torch.tensor(norm["mean"], device=device)[None]
        self.std = torch.tensor(norm["std"], device=device)[None]

    @staticmethod
    def _trellis2_config():
        from huggingface_hub import hf_hub_download
        return hf_hub_download(TRELLIS2, "pipeline.json")

    @classmethod
    def from_pretrained(cls, model="terzi_masked", repo=REPO, ss_path=None, slat_path=None, device="cuda", **kw):
        """model: terzi | terzi_masked | terzi_student | terzi_masked_student (folders of the Hugging Face repo), or local
        checkpoints with ss_path / slat_path (.safetensors or .pt state dicts with 'view_embed' and 'denoiser.*')."""
        from trellis2.models import ElasticSLatFlowModel, SparseStructureFlowModel
        if model not in MODELS:
            raise ValueError(f"unknown model {model}; one of {list(MODELS)}")
        masked, student = MODELS[model]
        if ss_path is None or slat_path is None:
            from huggingface_hub import hf_hub_download
            ss_path = ss_path or hf_hub_download(repo, f"{model}/ss.safetensors")
            slat_path = slat_path or hf_hub_download(repo, f"{model}/slat.safetensors")

        def load(cls_, args, path):
            sd = _state_dict(path)
            m = cls_(**args)
            miss, unexp = m.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("denoiser.")}, strict=False)
            assert set(miss) <= {"rope_phases"} and not unexp, (miss, unexp)
            return m.to(device).eval(), sd["view_embed"].to(device)
        ss, ve_ss = load(SparseStructureFlowModel, SS_ARGS, ss_path)
        slat, ve_slat = load(ElasticSLatFlowModel, SLAT_ARGS, slat_path)
        assert (len(ve_ss) == 6) == masked, "checkpoint and model name disagree on the garment mask input"
        return cls(ss, slat, ve_ss, ve_slat, masked, student, device=device, **kw)

    @torch.no_grad()
    def __call__(self, image, person_mask, K, body: Body, garment_mask=None, seed=0, holes="patch",
                 ss_steps=None, slat_steps=None, ss_cfg=None, slat_cfg=None, mask_crop=None, box=None):
        """image (H, W, 3) uint8, person_mask (H, W) uint8/bool, K (3, 3) intrinsics of the image, body in the camera frame.
        garment_mask (H, W): the garment(s) to generate (masked models only; the full-outfit mask generates the whole outfit),
        or mask_crop: the same mask already in the 512 x 512 crop (bool).
        box: crop box {cx, cy, size} in image pixels (default: the projected body, squared, padded by 1.2).
        seed: int or str; the same seed gives the same garment.
        holes: "patch" (default; needs body.joint_of_vertex, i.e. an SMPL-X body loaded with the model file), "fast" or "none"."""
        dev = self.device
        K = np.asarray(K, np.float64)
        image = np.asarray(image)[..., :3].astype(np.uint8)
        person_mask = np.asarray(person_mask)
        person_mask = person_mask.astype(np.uint8) * 255 if person_mask.dtype == bool else person_mask.astype(np.uint8)
        if self.masked and garment_mask is None and mask_crop is None:
            raise ValueError("the masked models need a garment mask (use the full-outfit mask for the whole outfit)")
        box = box or C.crop_box(body.vertices, K)
        cond = C.crop_rgba(image, person_mask, box)
        z_pel = float(body.pelvis[2])
        v_cube = C.cam_to_cube(body.vertices, box, K, z_pel)
        bv = torch.tensor(v_cube, dtype=torch.float32, device=dev)
        bf = torch.tensor(body.faces, dtype=torch.int32, device=dev)
        views = render_views(bv, bf)
        mask_np = None
        if self.masked:
            mask_np = np.asarray(mask_crop, bool) if mask_crop is not None else C.crop_mask(garment_mask, box)
            if mask_np.mean() < 0.005:
                raise ValueError(f"the garment mask covers only {100 * mask_np.mean():.2f}% of the crop")
            views = torch.cat([views, torch.from_numpy(mask_np).float().to(dev)[None, None].expand(1, 3, -1, -1)])
        tok = self.dino(torch.cat([_bake(cond)[None].to(dev), views]))

        ssp, slp = dict(self.ss_params), dict(self.slat_params)
        for p, st, cf in ((ssp, ss_steps, ss_cfg), (slp, slat_steps, slat_cfg)):
            if st:
                p["steps"] = st
            if cf is not None:
                p["guidance_strength"] = cf
        # Stage 1: sparse structure -> occupied cells of the 32^3 grid
        _seed(seed, "ss")
        c_ss = _cond_tokens(tok, self.ve_ss)
        zn = torch.randn(1, 8, 16, 16, 16, device=dev)
        z = self.ss_sampler.sample(self.ss, zn, cond=c_ss, neg_cond=torch.zeros_like(c_ss), **ssp, verbose=False).samples
        occ = self.ss_decoder(z) > 0
        coords = torch.argwhere(F.max_pool3d(occ.float(), 2, 2) > 0.5)[:, [0, 2, 3, 4]].int()
        if len(coords) == 0:
            raise RuntimeError("the structure stage produced no occupied cells")
        # Stage 2: structured latent on these cells -> garment mesh
        from trellis2.modules import sparse as sp
        _seed(seed, "garment")
        c_sl = _cond_tokens(tok, self.ve_slat)
        noise = sp.SparseTensor(feats=torch.randn(coords.shape[0], self.slat.out_channels, device=dev), coords=coords)
        zg = self.slat_sampler.sample(self.slat, noise, cond=c_sl, neg_cond=torch.zeros_like(c_sl), **slp, verbose=False).samples
        gm = self.decoder(zg.replace(zg.feats * self.std + self.mean))[0]

        g_cube = trimesh.Trimesh(gm.vertices.float().cpu().numpy(), gm.faces.cpu().numpy(), process=False)
        garment = trimesh.Trimesh(C.cube_to_cam(g_cube.vertices, box, K, z_pel), g_cube.faces, process=False)
        body_m = trimesh.Trimesh(C.cube_to_cam(v_cube.astype(np.float32), box, K, z_pel), body.faces, process=False)
        if holes in ("fast", "patch"):
            garment = fill_holes(garment)
        if holes == "patch":
            if body.joint_of_vertex is None:
                print("warning: patch hole filling needs an SMPL-X body loaded with the model file (load_body(..., smplx_model=...)); "
                      "using the fast hole filling only")
            else:
                garment = patch_holes(garment, body_m, body.joint_of_vertex)
        elif holes not in ("fast", "none"):
            raise ValueError(f"holes must be patch, fast or none, not {holes}")
        bv_img = (views[:4].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        return TerziOutput(garment, body_m, g_cube, cond, bv_img, mask_np, box)
