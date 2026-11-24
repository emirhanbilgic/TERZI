"""The conditioning body: an SMPL-X mesh in the camera frame (metres; x right, y down, z forward), as produced by human mesh
recovery methods such as GVHMR, PromptHMR or SAM 3D Body (paper Sec. 3)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_SMPLX = 10475


@dataclass
class Body:
    vertices: np.ndarray            # (N, 3) camera frame, metres
    faces: np.ndarray               # (M, 3)
    pelvis: np.ndarray              # (3,) camera frame, metres; its depth anchors the cube
    joint_of_vertex: np.ndarray = None  # (10475,) SMPL-X joint of each vertex (skinning argmax); needed by the patch hole filling


def _smplx_model(path):
    """SMPL-X model file (.npz or .pkl, from https://smpl-x.is.tue.mpg.de) -> dict with 'f', 'J_regressor' and 'weights'."""
    path = Path(path)
    if path.is_dir():
        path = path / "SMPLX_NEUTRAL.npz"
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        return {"f": np.asarray(d["f"], np.int64), "J_regressor": np.asarray(d["J_regressor"], np.float64), "weights": np.asarray(d["weights"])}
    import pickle
    d = pickle.load(open(path, "rb"), encoding="latin1")
    J = d["J_regressor"]
    J = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
    return {"f": np.asarray(d["f"], np.int64), "J_regressor": np.asarray(J, np.float64), "weights": np.asarray(d["weights"])}


def load_body(path=None, vertices=None, faces=None, pelvis=None, smplx_model=None):
    """Body from a file or arrays.

    path: .npz with 'vertices' (and optionally 'faces', 'pelvis', 'joints' with the pelvis first), or a mesh file (.obj, .ply,
    .glb). Faces missing (e.g. an .npz with vertices only) are taken from the SMPL-X model file. The pelvis, if not given, is
    the SMPL-X pelvis joint (needs the model file); for other topologies the centre of the body's bounding box is used.
    """
    if path is not None:
        path = Path(path)
        if path.suffix == ".npz":
            d = np.load(path, allow_pickle=True)
            vertices = d["vertices"]
            faces = d["faces"] if "faces" in d.files else faces
            if pelvis is None and "pelvis" in d.files:
                pelvis = d["pelvis"]
            elif pelvis is None and "joints" in d.files:
                pelvis = d["joints"][0]
        else:
            import trimesh
            m = trimesh.load(str(path), force="mesh", process=False)
            vertices, faces = m.vertices, m.faces
    vertices = np.asarray(vertices, np.float64)
    model = _smplx_model(smplx_model) if smplx_model is not None and len(vertices) == N_SMPLX else None
    if faces is None:
        if model is None or len(vertices) != N_SMPLX:
            raise ValueError("the body has no faces: give a mesh file, an npz with 'faces', or --smplx-model for SMPL-X vertices")
        faces = model["f"]
    if pelvis is None:
        if model is not None and len(vertices) == N_SMPLX:
            pelvis = model["J_regressor"][0] @ vertices
        else:
            print("warning: no pelvis given; using the centre of the body's bounding box")
            pelvis = (vertices.min(0) + vertices.max(0)) / 2
    part = model["weights"].argmax(1) if model is not None else None
    return Body(vertices, np.asarray(faces, np.int64), np.asarray(pelvis, np.float64), part)


def body_from_smplx_params(params, smplx_model):
    """SMPL-X parameters (dict or .npz with betas, global_orient, body_pose, transl; optional left/right_hand_pose, jaw_pose,
    expression) -> Body, with the `smplx` package. Neutral model, 10 betas."""
    import smplx
    import torch
    p = dict(np.load(params, allow_pickle=True)) if isinstance(params, (str, Path)) else dict(params)
    t = {k: torch.as_tensor(np.asarray(v, np.float32)).reshape(1, -1) for k, v in p.items()
         if k in ("betas", "global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose", "jaw_pose", "expression")}
    m = smplx.create(str(smplx_model), model_type="smplx", gender="neutral", use_pca=False, flat_hand_mean=True,
                     num_betas=t["betas"].shape[1])
    out = m(**t)
    v = out.vertices[0].detach().numpy().astype(np.float64)
    return Body(v, np.asarray(m.faces, np.int64), out.joints[0, 0].detach().numpy().astype(np.float64),
                m.lbs_weights.detach().numpy().argmax(1))
