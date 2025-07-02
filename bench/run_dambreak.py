import time
import numpy as np
from _sph import PyWorld

for n in [1_000_000 * i for i in range(1, 11)]:
    w = PyWorld()
    start = time.time()
    w.step(1/60.0)
    t = time.time() - start
    print(f"Particles: {n}, step time: {t*1000:.3f} ms")
