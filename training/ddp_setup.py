"""Shared DDP bootstrap for the distillation scripts. MUST be imported before
anything that imports flex_gemm (i.e. before trellis2).

Why the flex_gemm step: concurrent writers corrupt flex_gemm's autotune cache JSON
(FLEX_GEMM_AUTOTUNE_CACHE_PATH) and crash every importer; with torchrun all ranks of
a job would share one file, so each rank gets its own copy.
"""
import os, random, shutil
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
DDP = WORLD > 1
IS_MAIN = RANK == 0

if DDP and os.environ.get("FLEX_GEMM_AUTOTUNE_CACHE_PATH"):
    base = os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"]
    mine = (base[:-5] if base.endswith(".json") else base) + f"_r{LOCAL_RANK}.json"
    if os.path.exists(base) and not os.path.exists(mine):
        try: shutil.copyfile(base, mine)     # warm start from the job's cache
        except OSError: pass
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = mine

import numpy as np, torch
if DDP:
    torch.cuda.set_device(LOCAL_RANK)       # before build_pipeline()'s .cuda()
    # rank 0 alone validates and writes ~25 GB of checkpoints while the others wait in the next all-reduce: the default
    # 10 min collective timeout is too short on slower filesystems (a validation + save step can take longer)
    import datetime
    torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MIN", "120"))))
# distinct streams per rank: t indices, noise, and shard resampling
random.seed(1234 + RANK); np.random.seed(1234 + RANK); torch.manual_seed(1234 + RANK)

def log(*a):
    if IS_MAIN: print(*a, flush=True)
