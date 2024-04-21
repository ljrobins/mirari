import taichi as ti
from .sdf import *
import numpy as np
from typing import Callable


@ti.pyfunc
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


@ti.data_oriented
class Scene:
    def __init__(self, objects: Callable):
        self.objects = objects
        self._n_objs = len(objects)
    
    @ti.pyfunc
    def _sdf(self, r):
        return [obj.sdf(r) for obj in ti.static(self.objects)]
    
    @ti.pyfunc
    def sdf(self, r):
        dists = self._sdf(r)
        min_dist = np.inf
        min_obj = self.objects[0]
        for i in ti.static(range(self._n_objs)):
            if dists[i] < min_dist:
                min_dist = dists[i]
                min_obj = self.objects[i]
        return [min_dist, min_obj]


@ti.pyfunc
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
    return wall, geometry


@ti.pyfunc
def scene_three_objs():
    box1 = Box(origin=ti.Vector([0.0, 0.0, 0.0]), 
               radii=ti.Vector([0.2, 0.3, 0.3]),
               rv=ti.Vector([0.1, 0.2, 0.3]))
    t1 = Torus(origin=ti.Vector([0.4, 0.4, 0.5]), 
                radii=ti.Vector([0.2, 0.1, 0.0]), 
                rv=ti.Vector([0.1, 0.2, 0.3]),
                material=Material(cs=1.0, a=0.2))
    
    sph1 = Sphere(origin=ti.Vector([0.0, 0.6, 0.0]),
                  radii=ti.Vector([0.2, 0.0, 0.0]), 
                  material=Material(cs=1.0, a=0.2))
    sph2 = Sphere(origin=ti.Vector([0.4, 0.1, 0.0]),
                  radii=ti.Vector([0.4, 0.0, 0.0]), 
                  material=Material(cs=1.0, a=0.001))
    return (t1, sph1, sph2, box1)

# @ti.data_oriented
# class Scene:
#     sphere1 = Sphere(origin=[0.0, 0.7, 0.0], radius=0.2, material=Material(cs=1.0, n=0.2))
#     sphere2 = Sphere(origin=[0.0, 0.2, 0.0], radius=0.2, material=Material(cs=1.0, n=0.2))
#     _n_objs = 2
    
#     def sdf(self, r):
#         self.sphere1.r = r
#         self.sphere2.r = r
#         return ti.min(*[self.sphere1.sdf(), self.sphere2.sdf()])

    # @ti.func
    # def sdf(self, r):
    #     min_ind = -1
    #     min_dist = np.inf
    #     i = 0
    #     for k,obj in ti.static(self.objects):
    #         obj.sdf(r)
    #     while i < self._n_objs:
    #         dist_i = self.objects[i].sdf(r)
    #         if dist_i < min_dist:
    #             min_dist = dist_i
    #             min_ind = i
    #         i += 1
    #     return min_dist

@ti.func
def scene_devel(r):
    box1 = box(origin=ti.Vector([0.0, 0.0, 0.0]), 
               scales=0.3*ti.Vector([1.0, 1.0, 1.0]), r=r)
    box2 = box(origin=ti.Vector([-0.25, 0.0, 0.0]), 
               scales=ti.Vector([0.5, 0.01, 0.2]), r=r)
    # box3 = box(origin=ti.Vector([0.25, 0.0, 0.0]), 
    #            scales=ti.Vector([0.5, 0.01, 0.2]), r=r)
    return ti.min(box1)
