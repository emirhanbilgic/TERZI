#!/usr/bin/env python3
"""Generate 3D garments from one image and an estimated SMPL-X body.

    # whole outfit (no garment mask needed)
    python infer.py --image examples/pexels_8052611/image.jpg --person-mask examples/pexels_8052611/person_mask.png \
        --body examples/pexels_8052611/body.npz --K examples/pexels_8052611/K.json --smplx-model data/SMPLX_NEUTRAL.npz --out output/

    # selected garment(s) with the masked model; --fast uses the one-step student (about 0.1 s per image)
    python infer.py ... --model terzi_masked --garment-mask examples/pexels_8052611/garment_mask.png [--fast]

Inputs
  --image          RGB image
  --person-mask    the person's silhouette (white = person); with --auto-mask it is computed with BiRefNet (briaai/RMBG-2.0)
  --body           the body in the CAMERA frame of the image (metres, x right, y down, z forward): an .npz with 'vertices' (+ optional
                   'faces', 'pelvis' or 'joints'), a mesh file, or SMPL-X parameters with --smplx-params
  --K / --focal    camera intrinsics of the image: a json/npy 3x3 matrix, or a focal length in pixels (principal point = image centre)
  --garment-mask   masked models only: white = the garment(s) to generate; the full-outfit mask generates the whole outfit
Outputs (in --out): garment.glb / garment.obj (camera frame, metres), body.glb, cond.png (the 512 x 512 crop), body_views.png
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from terzi import MODELS, TerziPipeline, body_from_smplx_params, load_body


def read_K(a, w, h):
    if a.K:
        p = Path(a.K)
        K = np.load(p) if p.suffix == ".npy" else json.load(open(p))
        K = K["K"] if isinstance(K, dict) else K
        return np.asarray(K, np.float64).reshape(3, 3)
    if a.focal is None:
        raise SystemExit("give the camera intrinsics: --K <json|npy> or --focal <pixels> (use the focal length your body estimator used)")
    return np.array([[a.focal, 0, w / 2], [0, a.focal, h / 2], [0, 0, 1]], np.float64)


def person_mask_birefnet(image):
    from trellis2.pipelines import rembg
    m = rembg.BiRefNet(model_name="briaai/RMBG-2.0")
    m.cuda()
    out = m(Image.fromarray(image))
    return np.asarray(out.getchannel(3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--person-mask", default=None)
    ap.add_argument("--auto-mask", action="store_true", help="compute the person mask with BiRefNet (gated model briaai/RMBG-2.0)")
    ap.add_argument("--body", default=None, help="body .npz / mesh in the camera frame")
    ap.add_argument("--smplx-params", default=None, help=".npz with SMPL-X parameters (betas, global_orient, body_pose, transl, ...)")
    ap.add_argument("--smplx-model", default=None, help="SMPL-X model file (SMPLX_NEUTRAL.npz): faces / pelvis of SMPL-X vertices, or --smplx-params")
    ap.add_argument("--K", default=None)
    ap.add_argument("--focal", type=float, default=None)
    ap.add_argument("--garment-mask", default=None)
    ap.add_argument("--model", default=None, choices=list(MODELS), help="default: terzi, or terzi_masked when --garment-mask is given")
    ap.add_argument("--fast", action="store_true", help="the one-step distilled student of the chosen model")
    ap.add_argument("--ss", default=None, help="local SS checkpoint instead of the Hugging Face weights")
    ap.add_argument("--slat", default=None, help="local SLat checkpoint instead of the Hugging Face weights")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--holes", choices=["patch", "fast", "none"], default="patch",
                    help="patch (default): fill holes below 10 cm, then close enclosed bare body regions (< 2%% of the body) with body-shaped "
                         "patches (needs --smplx-model); fast: only the holes below 10 cm (the setting of the paper's numbers)")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    model = a.model or ("terzi_masked" if a.garment_mask else "terzi")
    if a.fast and not model.endswith("_student"):
        model += "_student"
    if MODELS[model][0] and not a.garment_mask:
        raise SystemExit(f"{model} needs --garment-mask")
    image = np.asarray(Image.open(a.image).convert("RGB"))
    H, W = image.shape[:2]
    K = read_K(a, W, H)
    if a.smplx_params:
        if not a.smplx_model:
            raise SystemExit("--smplx-params needs --smplx-model")
        body = body_from_smplx_params(a.smplx_params, a.smplx_model)
    elif a.body:
        body = load_body(a.body, smplx_model=a.smplx_model)
    else:
        raise SystemExit("give the body: --body or --smplx-params")
    if a.person_mask:
        pmask = np.asarray(Image.open(a.person_mask).convert("L"))
    elif a.auto_mask:
        pmask = person_mask_birefnet(image)
    else:
        raise SystemExit("give --person-mask, or --auto-mask to compute it")
    gmask = np.asarray(Image.open(a.garment_mask).convert("L")) if a.garment_mask else None

    pipe = TerziPipeline.from_pretrained(model, ss_path=a.ss, slat_path=a.slat)
    t0 = time.time()
    out = pipe(image, pmask, K, body, garment_mask=gmask, seed=a.seed, holes=a.holes)
    print(f"{model}: {len(out.garment.vertices)} vertices, {len(out.garment.faces)} faces ({time.time() - t0:.1f} s)")

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    out.garment.export(o / "garment.glb")
    out.garment.export(o / "garment.obj")
    out.body.export(o / "body.glb")
    out.cond.save(o / "cond.png")
    Image.fromarray(np.concatenate(list(out.body_views), 1)).save(o / "body_views.png")
    if out.mask is not None:
        Image.fromarray(out.mask.astype(np.uint8) * 255).save(o / "garment_mask_crop.png")
    print(f"saved to {o}/")


if __name__ == "__main__":
    main()
