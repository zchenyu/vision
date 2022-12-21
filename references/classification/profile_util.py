import tqdm
from itertools import islice
from PIL import Image
import numpy
import json
import time

def measure(name, dl, warmup=1):
    steps = 0
    rows = 0
    it = iter(dl)
    for _ in range(warmup):
        next(it)
    st = time.time()
    with tqdm.tqdm(desc=name, unit_scale=True, unit="rows") as t:
        tstart = time.perf_counter()
        for page in islice(it, 10000):
            # print('===',page[0].shape)
            if time.time() - st > 20:
                break
            if isinstance(page, (list, tuple)) and len(page) == 2:
                page = page[0]
            if isinstance(page, (torch.Tensor, numpy.ndarray)):
                cnt = page.shape[0] if len(page.shape) >= 4 else 1
            elif isinstance(page, (dict, Image.Image)):
                cnt = 1
            else:
                raise TypeError(f"Unknown type {type(page)}")
            steps += 1
            rows += cnt
            t.update(cnt)
        elapsed = time.perf_counter() - tstart
    print(f"{name} done === {steps} batches, {rows} total, {rows/elapsed:.0f} qps")

