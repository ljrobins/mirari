import taichi as ti


@ti.dataclass
class Material:
    cs: float
    a: float
    emmissive: bool
