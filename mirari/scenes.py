import taichi as ti
from .sdf import *
import numpy as np
from typing import Callable


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


@ti.data_oriented
class Scene:
    def __init__(self, objects: Callable):
        self.objects = objects
        self._n_objs = len(objects)
    
    @ti.func
    def _sdf(self, r):
        return [obj.sdf(r) for obj in ti.static(self.objects)]
    
    @ti.func
    def sdf(self, r):
        dists = self._sdf(r)
        min_dist = np.inf
        min_obj = self.objects[0]
        for i in ti.static(range(self._n_objs)):
            if dists[i] < min_dist:
                min_dist = dists[i]
                min_obj = self.objects[i]
        return [min_dist, min_obj]

def cornell_box_scene():
    diff_material = Material(cs=0.0, a=0.01)
    spec_material = Material(cs=1.0, a=0.01)
    light_material = Material(cs=25.0, emmissive=True)

    floor = Box(origin=ti.Vector([0,-1,0]),
                   radii=ti.Vector([1, 0.01, 2]),
                   material=diff_material)
    ceil = Box(origin=ti.Vector([0,1,0]),
                   radii=ti.Vector([1, 0.01, 2]),
                   material=diff_material)
    lwall = Box(origin=ti.Vector([-1,0,0]),
                   radii=ti.Vector([0.01, 1, 2]),
                   material=diff_material)
    rwall = Box(origin=ti.Vector([1,0,0]),
                   radii=ti.Vector([0.01, 1, 2]),
                   material=diff_material)
    bwall = Box(origin=ti.Vector([0,0,-1]),
                   radii=ti.Vector([1, 1, 0.01]),
                   material=diff_material)
    
    light = Box(
        origin=ti.Vector([0,0.99,0]),
        radii=ti.Vector([0.3, 0.01, 0.3]),
        material=light_material
    )

    box_back = Box(
        origin=ti.Vector([0.4,-0.5,-0.5]),
        radii=ti.Vector([0.3, 0.6, 0.3]),
        material=diff_material,
        rv=ti.Vector([0.0, 0.5, 0.0]),
    )

    box_front = Box(
        origin=ti.Vector([-0.3,-0.7,0.5]),
        radii=ti.Vector([0.3, 0.3, 0.3]),
        material=diff_material,
        rv=ti.Vector([0.0, -0.2, 0.0]),
    )

    s1 = Sphere(origin=ti.Vector([0.0, 0.0, 0.0]),
                radii=ti.Vector([0.2, 0.0, 0.0]), 
                material=spec_material
                )

    return (light,floor,ceil,lwall,rwall,bwall,box_front,box_back,s1)

def simple_scene():
    spec_material = Material(cs=1.0, a=0.01)
    light_material = Material(cs=25.0, emmissive=True)
    
    light = Box(
        origin=ti.Vector([0,0.99,0]),
        radii=ti.Vector([0.3, 0.01, 0.3]),
        material=light_material
    )

    s1 = Sphere(origin=ti.Vector([0.0, 0.0, 0.0]),
                radii=ti.Vector([0.2, 0.0, 0.0]), 
                material=spec_material
                )

    return (light,s1)


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
               rv=ti.Vector([0.1, 0.2, 0.3]),
               material=Material(cs=1.0, a=0.01))
    t1 = Torus(origin=ti.Vector([0.4, 0.4, 0.5]), 
                radii=ti.Vector([0.2, 0.1, 0.0]), 
                rv=ti.Vector([0.1, 0.2, 0.3]),
                material=Material(cs=1.0, a=0.2))
    
    sph1 = Sphere(origin=ti.Vector([0.0, 0.6, 0.0]),
                  radii=ti.Vector([0.2, 0.0, 0.0]), 
                  material=Material(cs=1.0, a=0.2, emmissive=True))
    sph2 = Sphere(origin=ti.Vector([0.4, 0.1, 0.0]),
                  radii=ti.Vector([0.4, 0.0, 0.0]), 
                  material=Material(cs=1.0, a=0.001))
    return (t1, sph1, sph2, box1)

@ti.pyfunc
def box_scene():
    b1 = Box(origin=ti.Vector([0.3, 0.0, 0.0]), 
               radii=ti.Vector([0.01, 0.3, 0.3]),
               material=Material(cs=1.0, a=0.01))

    b2 = Box(origin=ti.Vector([-0.3, 0.0, 0.0]), 
               radii=ti.Vector([0.01, 0.3, 0.3]),
               material=Material(cs=1.0, a=0.01, emmissive=True))

    b3 = Box(origin=ti.Vector([0.0, 0.0, 0.0]), 
               radii=ti.Vector([0.3, 0.01, 0.3]),
               material=Material(cs=1.0, a=0.01))

    s1 = Sphere(origin=ti.Vector([0.0, 0.6, 0.0]),
                  radii=ti.Vector([0.2, 0.0, 0.0]), 
                  material=Material(cs=1.0, a=0.2, emmissive=True))
    return (b1, b2, b3, s1)