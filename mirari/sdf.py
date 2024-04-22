import taichi as ti

from .math import rv_to_dcm
from .material import Material

@ti.dataclass
class Box:
    # USED
    origin: ti.math.vec3
    radii: ti.math.vec3
    rv: ti.math.vec3
    material: Material

    @ti.pyfunc
    def sdf(self, r: ti.math.vec3) -> float:
        rmo = r - self.origin
        if self.rv.norm() > 0.0:
            rmo = rv_to_dcm(-self.rv) @ rmo

        q = ti.abs(rmo) - self.radii
        return ti.Vector(
            [ti.max(0, q[0]), ti.max(0, q[1]), ti.max(0, q[2])]
        ).norm() + ti.min(q.max(), 0)

@ti.dataclass
class Torus:
    # USED
    origin: ti.math.vec3
    radii: ti.math.vec3
    rv: ti.math.vec3
    material: Material

    @ti.pyfunc
    def sdf(self, r: ti.math.vec3) -> float:
        rmo = r - self.origin
        if self.rv.norm() > 0.0:
            rmo = rv_to_dcm(-self.rv) @ rmo
        q = ti.Vector([
            ti.Vector([
            rmo[0], rmo[2]
        ]).norm() - self.radii[0],
        rmo[1]
        ])
        return q.norm() - self.radii[1]
    



@ti.dataclass
class Sphere:
    # USED
    origin: ti.math.vec3
    radii: ti.math.vec3
    material: Material

    # UNUSED
    rv: ti.math.vec3

    @ti.pyfunc
    def sdf(self, r):
        return (r - self.origin).norm() - self.radii[0]

