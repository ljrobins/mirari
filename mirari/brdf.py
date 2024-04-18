import taichi as ti

from .math import slerp, random_direction


@ti.func
def out_dir(d, n):
    spec_dir = (d - 2 * d.dot(n) * n).normalized()
    lamb_dir = (n + random_direction()).normalized()
    return slerp(lamb_dir, spec_dir, 0.94)
