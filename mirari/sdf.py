import taichi as ti
import numpy as np

from .math import rv_to_dcm

@ti.func
def box(origin: ti.math.vec3, 
        scales: ti.math.vec3, 
        r: ti.math.vec3) -> float:
    q = ti.abs(r - origin) - scales
    return ti.Vector(
        [ti.max(0, q[0]), ti.max(0, q[1]), ti.max(0, q[2])]
    ).norm() + ti.min(q.max(), 0)

@ti.func
def torus(origin: ti.math.vec3,
          radii: ti.math.vec2, 
          r: ti.math.vec3,
          rv = ti.math.vec3([0, 0, 0])) -> float:
    rmo = r - origin
    if rv.norm() > 0:
        rmo = rv_to_dcm(-rv) @ rmo
    q = ti.Vector([
        ti.Vector([
        rmo[0], rmo[2]
    ]).norm() - radii[0],
    rmo[1]
    ])
    return q.norm() - radii[1]

@ti.func
def sphere(origin: ti.math.vec3, radius: float, r: ti.math.vec3):
    return (r - origin).norm() - radius