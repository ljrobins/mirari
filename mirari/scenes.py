import taichi as ti
from .sdf import *


@ti.func
def make_nested(f):
    f = f * 40
    i = int(f)
    if f < 0:
        if i % 2 == 1:
            f -= ti.floor(f)
        else:
            f = ti.floor(f) + 1 - f
    f = (f - 0.2) / 40
    return f


@ti.func
def scene_one(o):
    wall = ti.min(o[1] + 0.1, o[2] + 0.4)
    sphere = (o - ti.Vector([0.0, 0.35, 0.0])).norm() - 0.36

    q = ti.abs(o - ti.Vector([0.8, 0.3, 0])) - ti.Vector([0.3, 0.3, 0.3])
    box = ti.Vector(
        [ti.max(0, q[0]), ti.max(0, q[1]), ti.max(0, q[2])]
    ).norm() + ti.min(q.max(), 0)

    O = o - ti.Vector([-0.8, 0.3, 0])
    d = ti.Vector([ti.Vector([O[0], O[2]]).norm() - 0.3, abs(O[1]) - 0.3])
    cylinder = (
        ti.min(d.max(), 0.0) + ti.Vector([ti.max(0, d[0]), ti.max(0, d[1])]).norm()
    )

    geometry = make_nested(ti.min(sphere, box, cylinder))
    geometry = ti.max(geometry, -(0.32 - (o[1] * 0.6 + o[2] * 0.8)))
    return ti.min(wall, geometry)

@ti.func
def scene_two(r):
    sphere = (r - ti.Vector([0.0, 0.35, 0.0])).norm() - 0.36
    sphere2 = (r - ti.Vector([0.0, 0.7, 0.0])).norm() - 0.2
    return ti.min(sphere, sphere2)

@ti.func
def scene_three(r):
    box1 = box(origin=ti.Vector([0.0, 0.0, 0.0]), 
               scales=ti.Vector([0.2, 0.3, 0.3]), r=r)
    t1 = torus(origin=ti.Vector([0.4, 0.4, 0.5]), 
                radii=ti.Vector([0.2, 0.1]), 
                r=r,
                rv=ti.Vector([0.1, 0.2, 0.3]))
    sph1 = sphere(origin=ti.Vector([0.0, 0.6, 0.0]),
                  radius=0.2, r=r)
    return ti.min(box1, t1, sph1)

@ti.func
def scene_devel(r):
    box1 = box(origin=ti.Vector([0.0, 0.0, 0.0]), 
               scales=0.3*ti.Vector([1.0, 1.0, 1.0]), r=r)
    # box2 = box(origin=ti.Vector([-0.25, 0.0, 0.0]), 
    #            scales=ti.Vector([0.5, 0.01, 0.2]), r=r)
    # box3 = box(origin=ti.Vector([0.25, 0.0, 0.0]), 
    #            scales=ti.Vector([0.5, 0.01, 0.2]), r=r)
    return ti.min(box1)
