"""Image-aligned garment space (paper Sec. 3.1.1, supplementary Sec. 6.1).

The front face of the cube [-1/2, 1/2]^3 is the 512 x 512 input crop: a vertex lies on the crop pixel it projects to, and its
third coordinate is its depth relative to the pelvis, in crop pixels, times a fixed gain of 0.6. The mapping is invertible in
closed form given the crop box, the camera intrinsics K and the pelvis depth.
"""
import cv2
import numpy as np
from PIL import Image

CROP = 512                        # crop resolution = cube extent in crop pixels
Z_GAIN = 0.6                      # depth gain: depth half-extent of the cube = 512 / 1.2 crop pixels
INV_SCALE_UV = float(CROP)        # u_crop = x * 512 + 256
INV_SCALE_Z = float(CROP) / Z_GAIN
PAD = 1.2                         # crop = projected body box, squared, padded by 1.2


def project(points, K):
    """camera-frame points (N, 3) in metres -> pixel coordinates u, v and a validity mask (in front of the camera)."""
    P = np.asarray(points, np.float64)
    ok = P[:, 2] > 0.01
    u = np.full(len(P), -1.0)
    v = np.full(len(P), -1.0)
    u[ok] = K[0, 0] * P[ok, 0] / P[ok, 2] + K[0, 2]
    v[ok] = K[1, 1] * P[ok, 1] / P[ok, 2] + K[1, 2]
    return u, v, ok


def crop_box(body_vertices, K, pad=PAD):
    """square box around the projected body, padded by `pad`: {cx, cy, size} in image pixels."""
    u, v, ok = project(body_vertices, K)
    if ok.sum() < 10:
        raise ValueError("the body is not in front of the camera")
    u, v = u[ok], v[ok]
    return {"cx": float((u.min() + u.max()) / 2), "cy": float((v.min() + v.max()) / 2),
            "size": float(max(u.max() - u.min(), v.max() - v.min()) * pad)}


def crop_rgba(image, person_mask, box, res=CROP):
    """RGB image (H, W, 3) uint8 and person mask (H, W) uint8 -> res x res RGBA crop of `box` (zero padding outside the image)."""
    cx, cy, size = box["cx"], box["cy"], box["size"]
    x1, y1 = int(np.floor(cx - size / 2)), int(np.floor(cy - size / 2))
    x2, y2 = int(np.ceil(cx + size / 2)), int(np.ceil(cy + size / 2))
    H, W = image.shape[:2]
    if x2 < 0 or x1 > W or y2 < 0 or y1 > H:
        raise ValueError("the crop box lies outside the image")
    ix1, iy1, ix2, iy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    pl, pt, pr, pb = ix1 - x1, iy1 - y1, x2 - ix2, y2 - iy2
    rgb = cv2.copyMakeBorder(image[iy1:iy2, ix1:ix2], pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    msk = cv2.copyMakeBorder(person_mask[iy1:iy2, ix1:ix2], pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    rgb = cv2.resize(rgb, (res, res), cv2.INTER_LINEAR)
    msk = cv2.resize(msk, (res, res), cv2.INTER_LINEAR)
    return Image.fromarray(np.dstack([rgb, msk]).astype(np.uint8), mode="RGBA")


def crop_mask(mask, box, res=CROP):
    """mask (H, W), bool or uint8 0..255 -> res x res boolean crop of `box`, resampled exactly like the RGB crop."""
    m = np.asarray(mask)
    m = m.astype(np.uint8) * 255 if m.dtype == bool else m.astype(np.uint8)
    rgba = crop_rgba(np.dstack([m] * 3), m, box, res)
    return np.asarray(rgba.getchannel(3)) > 127


def cam_to_cube(points, box, K, z_pelvis, res=CROP):
    """camera-frame metres -> cube coordinates [-1/2, 1/2]^3."""
    P = np.asarray(points, np.float64)
    u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
    v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
    k = res / box["size"]
    uc = (u - (box["cx"] - box["size"] / 2)) * k
    vc = (v - (box["cy"] - box["size"] / 2)) * k
    zc = (P[:, 2] - z_pelvis) * (K[0, 0] * k) / z_pelvis
    return np.stack([(uc - res / 2) / INV_SCALE_UV, (vc - res / 2) / INV_SCALE_UV, zc / INV_SCALE_Z], 1)


def cube_to_cam(points, box, K, z_pelvis, res=CROP):
    """inverse of cam_to_cube: cube coordinates -> camera-frame metres."""
    X = np.asarray(points, np.float64)
    u_img = (box["cx"] - box["size"] / 2) + (X[:, 0] * INV_SCALE_UV + res / 2) * box["size"] / res
    v_img = (box["cy"] - box["size"] / 2) + (X[:, 1] * INV_SCALE_UV + res / 2) * box["size"] / res
    f_crop = float(K[0, 0]) * res / box["size"]
    Z = z_pelvis * (1.0 + X[:, 2] * INV_SCALE_Z / f_crop)
    return np.stack([(u_img - K[0, 2]) * Z / K[0, 0], (v_img - K[1, 2]) * Z / K[1, 1], Z], 1)
