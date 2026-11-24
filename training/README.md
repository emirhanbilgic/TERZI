# Training

The released models were trained in five stages (paper Secs. 3.2-3.4, supplementary Secs. 6-7). All of them run from the
repository root; multi-GPU stages use `torchrun`. Paths default to the layout below and can be changed on the command line.

```
data/bedlam2/                 BEDLAM2.0 (bedlam2.is.tue.mpg.de): sqlite + gt_mapping.json, motions, clothing NPZs, MP4s, camera CSVs,
                              b2_clothing_graded_apose_obj/ (A-pose garments + their bodies)
data/SMPLX_NEUTRAL.npz        SMPL-X model
two_branch_data/              whole-outfit training data (stage 1)
pg/                           per-garment and A-pose data (stage 3)
full/index.pkl                index of the masked model's data
```

## 1. Whole-outfit data (`training/data/`)

Every BEDLAM2.0 single-person sequence with clothing simulation, 40 frames each (642,280 posed samples), kept when the
reprojected person lies fully inside the image and fits the cube in depth (510,271 samples: 507,940 train, 2,331 validation).

```bash
# posed SMPL-X body + simulated garments per frame
python training/data/build_posed_dataset.py --data-dir data/bedlam2 --smplx-model data/SMPLX_NEUTRAL.npz \
    --output-dir posed --frames-per-sequence 40 --garments-only --testset train          # and --testset test
# image-aligned crops and meshes in the cube [-1/2, 1/2]^3 (crop_space.py), body and garment branches
python training/data/reproject_weak_perspective.py --input_dir posed --data_dir data/bedlam2 --output_dir reproj_body \
    --branch body --resolution 512 --crop_mode bbox --views 1 --no-glb --require_full_frame --require_no_z_clip
python training/data/reproject_weak_perspective.py --input_dir posed --data_dir data/bedlam2 --output_dir reproj_garment \
    --branch garment --resolution 512 --crop_mode bbox --views 1 --no-glb --require_full_frame --require_no_z_clip
# structured latents (TRELLIS.2 shape encoder, 512) and sparse-structure latents; --shard / --total_shards to parallelise
python training/data/prepare_bedlam_dit_data.py --reprojection_dir reproj_body --output_dir two_branch_data/shape_body_aligned --grid_size 512 --skip_slat_stats
python training/data/prepare_bedlam_dit_data.py --reprojection_dir reproj_garment --output_dir two_branch_data/shape_garment_aligned --grid_size 512 --skip_slat_stats
python training/data/finalize_shard_metadata.py --dir two_branch_data/shape_garment_aligned --subdirs shape_latent render_cond
python training/data/prepare_bedlam_ss_data.py --dit_data_dir two_branch_data/shape_garment_aligned --reprojection_dir reproj_garment \
    --output_dir two_branch_data/ss_garment_aligned                                        # likewise for the body
# WebDataset shards (5% validation split, seed 0)
for b in body garment; do
  python training/data/webdataset_convert.py --mode shape --shape_dir two_branch_data/shape_${b}_aligned \
      --render_cond two_branch_data/shape_body_aligned/render_cond --out_dir two_branch_data/wds/shape_${b}_aligned --prefix shape_${b}
done
python training/data/webdataset_convert.py --mode ss --ss_body two_branch_data/ss_body_aligned --ss_garment two_branch_data/ss_garment_aligned \
    --render_cond two_branch_data/shape_body_aligned/render_cond --out_dir two_branch_data/wds/ss_aligned --prefix ss
```

`prepare_bedlam_dit_data.py` / `prepare_bedlam_ss_data.py` also extract the pretrained TRELLIS.2 flow models used as the
initialisation (`pretrained_denoiser.pt`, `pretrained_ss_flow.pt`; skip with `--skip_denoiser_extract` / `--skip_flow_extract`).

## 2. Main model (`train_main.py`)

RGB crop + four body renders; image and body always from the same frame. 16 GPUs in the paper (2 nodes x 8; supplementary
Tab. 5); the released checkpoints are SS step 24,900 and SLat step 87,000.

```bash
torchrun --nnodes=2 --nproc_per_node=8 training/train_main.py --kind ss   --micro-bs 8  --out results/main_ss
torchrun --nnodes=2 --nproc_per_node=8 training/train_main.py --kind slat --micro-bs 12 --out results/main_slat
```

## 3. Masked model (`train_masked.py`)

Adds the garment mask as a sixth view and is initialised from the main model. Every sample serves the whole outfit and each
visible garment; the image comes from the same frame (50%), another frame of the same sequence (40%) or the A-pose frame
(10%), while the body and the target garment always come from the conditioning frame.

```bash
# per-garment split, masks and A-pose garments (+ their encodings with prepare_bedlam_*_data.py, as in stage 1)
python training/data/pergarment_prep.py --src reproj_garment --out pg/pergarment --ids ids.txt
python training/data/apose_map.py --bedlam2 data/bedlam2 --ids splits.json --out apose_map.json
python training/data/apose_prep.py --src reproj_garment --map apose_map.json --ids ids.txt --out pg/apose --per-garment
python training/data/full_index.py --splits splits.json --root pg/pergarment,pg/apose,ids.txt --out full/index.pkl
torchrun --nproc_per_node=4 training/train_masked.py --kind ss   --init <main SS ckpt>   --out results/masked_ss \
    --steps 60000 --grad-accum 2 --lr 1e-4 --warmup 500 --t-max 60000
torchrun --nproc_per_node=4 training/train_masked.py --kind slat --init <main SLat ckpt> --out results/masked_slat \
    --steps 100000 --grad-accum 3 --lr 2e-5 --warmup 500 --t-max 100000
```

The checkpoint with the best validation loss of each stage is kept (`ckpts/denoiser_best.pt`).

## 4. Collision-aware fine-tuning (`finetune/`)

Only the SLat DiT is fine-tuned, on the frozen model's own samples, with the anchor to a frozen copy and the penetration
penalty on the decoded garment (clearance 1.5 crop px, weight 30, penalty for t <= 0.6; supplementary Tab. 6). The released
teachers are the checkpoints after 100 steps (800 examples).

```bash
# masked model: its own samples of every garment and the whole outfit (+ condition tokens and the GT body)
python training/finetune/datagen_masked.py --ss <masked SS> --slat <masked SLat> --split train --out ft/masked/train   # and --split val
torchrun --nproc_per_node=1 training/finetune/finetune_masked.py --data ft/masked/train --val-data ft/masked/val \
    --init <masked SLat> --out results/ft_masked --keep-steps
# main model: its own whole-outfit samples on the same outfits (tokens from datagen_masked.py --tokens-only)
python training/finetune/datagen_main.py --ids train_ids.txt --layers ft/masked/train --ss <main SS> --slat <main SLat> --out ft/main/train
torchrun --nproc_per_node=1 training/finetune/finetune_main.py --pairs ft/main/train --val-pairs ft/main/val \
    --init <main SLat> --out results/ft_main --keep-steps
```

## 5. One-step students (`distill/distill.py`)

Per stage and per teacher: reflow on the teacher's own samples, then one-step distillation of the reflowed model (SLat
6,000 + 6,000 updates, SS 3,000 + 3,000; 48,665 training outfits with cached condition tokens). See the docstring of
`distill.py` for the two commands.

## Checkpoints

Every script saves `{denoiser.*, view_embed}` state dicts, the format `terzi.TerziPipeline.from_pretrained(model, ss_path=...,
slat_path=...)` loads (`.pt` or `.safetensors`).
