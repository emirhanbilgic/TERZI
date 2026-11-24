#!/usr/bin/env python3
"""
Image-aligned crop space — the ONE definition of how an image crop maps to the
[-0.5, 0.5]^3 cube that the SC-VAE / SS-VAE / DiT operate in.

THE CONTRACT
------------
    The cube's XY face IS the crop image.

        crop pixel (0, 0)      <->  (x, y) = (-0.5, -0.5)
        crop pixel (S/2, S/2)  <->  (x, y) = ( 0.0,  0.0)
        crop pixel (S, S)      <->  (x, y) = (+0.5, +0.5)

    Every pixel of the target therefore corresponds to the exact same location
    as in the conditioning image. A vertex that projects OUTSIDE the crop lands
    OUTSIDE the cube and is clipped away — so a cropped image yields a cropped
    mesh, pixel for pixel. Z is measured in the same crop pixels, anchored at
    the pelvis depth, with a fixed gain (see Z_GAIN) that gives the cube enough
    depth for a human, so the cube is a genuine (weak-perspective) 3D box
    sitting in front of the camera.

WHAT THIS REPLACES (the alignment bug)
--------------------------------------
The previous mapping was

    x = (u_crop - 256) * 1.2/512      y = (v_crop - 256) * 1.2/512
    z =  z_crop * 0.5/800

with the crop box itself defined as "the projected bbox of THIS mesh, squared
and padded by 1.2". Composed, that is an OBJECT-centric normalisation: whatever
the framing, the mesh was re-centred and re-scaled to fill the cube. The target
could not express "the person is cut off at the knees" or "the person sits in
the left third of the frame" — every target was a full, centred, canonically
scaled human. The conditioning image was then independently re-cropped to the
*alpha* bounding box (ImageConditionedMixin.get_instance / bake_cond), so image
and target were only ever aligned up to the accident of the segmentation mask
matching the mesh bbox.

Here the crop box is chosen ONCE, from the image, and both the pixels and the
geometry are mapped through it. Nothing is re-centred, nothing is re-scaled.

FORWARD / INVERSE
-----------------
    box    = CropBox(cx, cy, size)         # square, in ORIGINAL image pixels
    scale  = out / size                    # out = crop resolution (512)

    u_crop = (u - cx + size/2) * scale     # full-perspective projection
    v_crop = (v - cy + size/2) * scale
    f_crop = fx * scale                    # focal length in crop pixels
    z_crop = (Z_v - Z_pelvis) * f_crop / Z_pelvis

    x = u_crop / out - 0.5                 # cube (this module's to_cube)
    y = v_crop / out - 0.5
    z = z_crop / out

    u_crop = (x + 0.5) * out               # inverse (from_cube)
    u_img  = u_crop / scale + cx - size/2  # back to the original image
    Z_v    = Z_pelvis * (1 + z_crop / f_crop)          # metric depth
    X_v    = (u_img - K[0,2]) * Z_v / K[0,0]           # metric camera space

Every producer and consumer of crop-space geometry must go through this module.
"""

from dataclasses import dataclass, asdict

import numpy as np

# Crop resolution. The cube spans exactly this many crop pixels on all 3 axes.
CROP_SIZE = 512

# Nudge clipped geometry just inside the cube instead of exactly on the face, so
# voxel indices can never round to grid_size. 1e-4 cube units = 0.05 crop px.
CUBE_EPS = 1e-4

# Depth scale: z = z_crop * Z_GAIN / CROP_SIZE, so the cube's Z half-extent is
# CROP_SIZE / (2 * Z_GAIN) crop pixels.
#
# Z_GAIN = 1.2/2 = 0.6 puts that half-extent at 512/1.2 = 426.67 crop px —
# exactly the span a bbox-framed subject occupies in the crop. So the cube is as
# deep as the subject is tall, which is the natural amount of room for a human.
#
# Why not 1.0 (isotropic with X/Y, half-extent 256)? Profiled over 400 BEDLAM2
# samples, pre-clip max |z_crop| is p50 99, p90 201, p99 329, max 398 crop px:
#     half-extent 256 (gain 1.00)  ->  4.00% of samples lose geometry
#     half-extent 341 (gain 0.75)  ->  0.25%
#     half-extent 427 (gain 0.60)  ->  0.00%
# and the live run confirmed it — 2.2% of 31k samples clipped on Z, the worst
# losing 64% of the body. This is NOT about tall subjects: the crop normalises
# apparent size, so correlation(height in crop, max |z_crop|) is r = -0.11. It
# is about poses pointing along the camera axis — reaching, lunging, leaning in
# — where the depth extent rivals the height.
#
# The cost is a known 1/0.6 = 1.667x stretch of Z relative to X/Y (the old
# pipeline's was 3.75x). Image alignment is an X/Y property and is untouched:
# the cube's XY face is still exactly the crop.
Z_GAIN = 1.2 / 2.0


# ── The crop box ─────────────────────────────────────────────────────────────

@dataclass
class CropBox:
    """A square crop of the original image, in ORIGINAL image pixels.

    cx, cy : centre of the square
    size   : side length
    out    : resolution the crop is resampled to (the cube's pixel span)
    """
    cx: float
    cy: float
    size: float
    out: int = CROP_SIZE

    @property
    def scale(self):
        """Crop pixels per original pixel."""
        return self.out / self.size

    @property
    def origin(self):
        """Top-left corner of the square in original image pixels."""
        return self.cx - self.size / 2.0, self.cy - self.size / 2.0

    def project(self, u, v):
        """Original image pixels -> crop pixels (exact, sub-pixel)."""
        ox, oy = self.origin
        return (u - ox) * self.scale, (v - oy) * self.scale

    def unproject(self, u_crop, v_crop):
        """Crop pixels -> original image pixels."""
        ox, oy = self.origin
        return u_crop / self.scale + ox, v_crop / self.scale + oy

    def focal_crop(self, f_px):
        """Focal length expressed in crop pixels."""
        return f_px * self.scale

    def image_rect(self, img_w, img_h):
        """The render's borders, in CROP pixels: (x0, y0, x1, y1).

        A square crop of a 16:9 render routinely reaches past the frame, and a
        subject can also walk out of shot. Everything beyond this rectangle is
        padding the camera never recorded, so the target must not contain
        geometry there — see clip_mesh_to_cube(visible_rect=...).
        """
        x0, y0 = self.project(0.0, 0.0)
        x1, y1 = self.project(float(img_w), float(img_h))
        return float(x0), float(y0), float(x1), float(y1)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return CropBox(float(d["cx"]), float(d["cy"]), float(d["size"]),
                       int(d.get("out", CROP_SIZE)))


def bbox_crop_box(u, v, valid, padding=1.2, out=CROP_SIZE):
    """Deterministic framing: square around the projected bbox, padded.

    This reproduces the framing the dataset has always used — only the mapping
    from that framing to the cube changes. Returns None if nothing is visible.
    """
    if not np.any(valid):
        return None
    uu, vv = u[valid], v[valid]
    cx = float((uu.min() + uu.max()) / 2.0)
    cy = float((vv.min() + vv.max()) / 2.0)
    extent = float(max(uu.max() - uu.min(), vv.max() - vv.min()))
    if not np.isfinite(extent) or extent <= 0:
        return None
    return CropBox(cx, cy, extent * padding, out)


def jitter_crop_box(box, rng, scale_range=(0.75, 1.5), shift_frac=0.15,
                    base_padding=1.2):
    """Randomised framing around a deterministic box.

    `box` is the padded bbox box; the person's tight extent is
    box.size / base_padding. The new side is drawn as

        size = extent * U(scale_range)

    so scale < 1.0 crops INTO the person (truncated head / legs / arms — the
    case the model has to learn to read), and scale > 1.0 pulls back and leaves
    real background around them. The centre is then shifted by up to
    ±shift_frac of the new side on each axis, which moves the person off-centre.

    Both effects are carried into the target by construction: the same box maps
    the pixels and the geometry, and whatever leaves the frame leaves the cube.
    """
    extent = box.size / base_padding
    size = float(extent * rng.uniform(*scale_range))
    cx = float(box.cx + rng.uniform(-shift_frac, shift_frac) * size)
    cy = float(box.cy + rng.uniform(-shift_frac, shift_frac) * size)
    return CropBox(cx, cy, size, box.out)


# ── Image resampling ─────────────────────────────────────────────────────────

def crop_image(img, box, border_value=0):
    """Resample `img` through `box` to (box.out, box.out).

    Uses a single affine warp, which is the exact pixel-space twin of
    CropBox.project: original pixel (u, v) lands at crop pixel
    ((u-ox)*scale, (v-oy)*scale). No floor/ceil/pad rounding, so image and
    geometry agree sub-pixel.

    Returns:
        crop    (out, out, 3) uint8 — border_value outside the original image
        inframe (out, out)    uint8 — 255 where the crop saw real pixels
    """
    import cv2

    ox, oy = box.origin
    s = box.scale
    M = np.array([[s, 0.0, -ox * s],
                  [0.0, s, -oy * s]], dtype=np.float64)
    crop = cv2.warpAffine(img, M, (box.out, box.out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(border_value,) * 3)
    ones = np.full(img.shape[:2], 255, np.uint8)
    inframe = cv2.warpAffine(ones, M, (box.out, box.out), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return crop, inframe


# ── Geometry: crop pixels <-> cube ───────────────────────────────────────────

def depth_to_crop(z_cam, z_pelvis, f_crop):
    """Camera-space depth (metres) -> pelvis-relative depth in crop pixels."""
    return (z_cam - z_pelvis) * f_crop / z_pelvis


def crop_to_depth(z_crop, z_pelvis, f_crop):
    """Inverse of depth_to_crop."""
    return z_pelvis * (1.0 + z_crop / f_crop)


def to_cube(vertices_crop, out=CROP_SIZE, z_gain=Z_GAIN):
    """(u_crop, v_crop, z_crop) -> [-0.5, 0.5]^3. Isotropic, no re-centring."""
    v = np.asarray(vertices_crop, np.float64)
    return np.stack([
        v[:, 0] / out - 0.5,
        v[:, 1] / out - 0.5,
        v[:, 2] / out * z_gain,
    ], axis=1).astype(np.float32)


def from_cube(vertices_cube, out=CROP_SIZE, z_gain=Z_GAIN):
    """[-0.5, 0.5]^3 -> (u_crop, v_crop, z_crop). Exact inverse of to_cube."""
    v = np.asarray(vertices_cube, np.float64)
    return np.stack([
        (v[:, 0] + 0.5) * out,
        (v[:, 1] + 0.5) * out,
        v[:, 2] * out / z_gain,
    ], axis=1).astype(np.float32)


# Affine form of to_cube, for code that wants constants rather than a call.
#   cube = (crop_px - NORM_CENTER) * NORM_SCALE
NORM_CENTER = np.array([CROP_SIZE / 2.0, CROP_SIZE / 2.0, 0.0], dtype=np.float32)
NORM_SCALE = np.array([1.0 / CROP_SIZE,
                       1.0 / CROP_SIZE,
                       Z_GAIN / CROP_SIZE], dtype=np.float32)
# Inverse scale, for de-normalising a prediction back to crop pixels.
INV_SCALE_UV = float(CROP_SIZE)
INV_SCALE_Z = float(CROP_SIZE) / Z_GAIN


def clip_mesh_to_cube(vertices_crop, faces, out=CROP_SIZE, z_gain=Z_GAIN,
                      visible_rect=None):
    """Clip a crop-space mesh to the cube — this is what makes a cropped input
    produce a cropped target.

    Cuts triangles exactly at the six faces (new vertices on the boundary)
    rather than dropping whole faces, so the silhouette follows the frame edge.
    The result is an open surface where it was cut; that is fine — everything
    downstream (flexible dual grid, SS occupancy) consumes a surface.

    visible_rect: (x0, y0, x1, y1) in crop pixels, from CropBox.image_rect —
    the part of the crop that shows real recorded pixels. Geometry beyond it is
    removed too, so the target never contains a limb the camera never saw. On
    BEDLAM2 this bites on ~21% of samples (up to 81% of one body).

    Returns (vertices, faces, stats). Faces may be empty if nothing is visible.
    """
    from trimesh.intersections import slice_faces_plane

    v = np.asarray(vertices_crop, np.float64)
    f = np.asarray(faces, np.int64)
    lo_xy = CUBE_EPS * out
    hi_xy = out - CUBE_EPS * out
    z_half = out / (2.0 * z_gain)
    lo_z, hi_z = -z_half + CUBE_EPS * out, z_half - CUBE_EPS * out

    # The cube and the recorded image are both hard limits; take the tighter.
    x_lo, x_hi, y_lo, y_hi = lo_xy, hi_xy, lo_xy, hi_xy
    if visible_rect is not None:
        rx0, ry0, rx1, ry1 = visible_rect
        x_lo, x_hi = max(x_lo, min(rx0, rx1)), min(x_hi, max(rx0, rx1))
        y_lo, y_hi = max(y_lo, min(ry0, ry1)), min(y_hi, max(ry0, ry1))

    n_before = len(f)
    inside = ((v[:, 0] >= x_lo) & (v[:, 0] <= x_hi) &
              (v[:, 1] >= y_lo) & (v[:, 1] <= y_hi) &
              (v[:, 2] >= lo_z) & (v[:, 2] <= hi_z))
    frac_outside = float(1.0 - inside.mean()) if len(v) else 1.0

    planes = [
        (np.array([1.0, 0, 0]), np.array([x_lo, 0, 0])),
        (np.array([-1.0, 0, 0]), np.array([x_hi, 0, 0])),
        (np.array([0, 1.0, 0]), np.array([0, y_lo, 0])),
        (np.array([0, -1.0, 0]), np.array([0, y_hi, 0])),
        (np.array([0, 0, 1.0]), np.array([0, 0, lo_z])),
        (np.array([0, 0, -1.0]), np.array([0, 0, hi_z])),
    ]
    for normal, origin in planes:
        if len(f) == 0:
            break
        # Skip planes the mesh does not actually cross (the common case).
        d = (v - origin) @ normal
        if d.min() >= 0:
            continue
        v, f = slice_faces_plane(v, f, plane_normal=normal, plane_origin=origin)[:2]

    stats = {
        "verts_outside_frac": frac_outside,
        "faces_before": int(n_before),
        "faces_after": int(len(f)),
        "truncated": bool(frac_outside > 0),
    }
    return np.asarray(v, np.float32), np.asarray(f, np.int64), stats
