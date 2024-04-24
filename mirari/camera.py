import taichi as ti
import numpy as np

@ti.dataclass
class Ray:
    position: ti.math.vec3
    direction: ti.math.vec3
    power: float

@ti.data_oriented
class Camera:

    def __init__(self, pos, dir, up, res, fov, is_perspective: bool):
        self.pos_field = ti.field(dtype=ti.f32, shape=(3,))
        self.dir_field = ti.field(dtype=ti.f32, shape=(3,))
        self.up_field = ti.field(dtype=ti.f32, shape=(3,))
        self.res = res
        self.res_vector = ti.Vector([*res])
        self.fov = fov
        self.is_perspective = is_perspective

        self.pos = pos.to_numpy()
        self.up = up.to_numpy()
        self.dir = dir.to_numpy()

    @ti.func
    def _pos(self):
        return ti.math.vec3(self.pos_field[0], self.pos_field[1], self.pos_field[2])

    @ti.func
    def _dir(self):
        return ti.math.vec3(self.dir_field[0], self.dir_field[1], self.dir_field[2])

    @ti.func
    def _up(self):
        return ti.math.vec3(self.up_field[0], self.up_field[1], self.up_field[2])

    @property
    def pos(self):
        return ti.Vector(
            [self.pos_field[0], self.pos_field[1], self.pos_field[2]]
        )

    @property
    def dir(self):
        return ti.Vector(
            [self.dir_field[0], self.dir_field[1], self.dir_field[2]]
        )

    @property
    def up(self):
        return ti.Vector(
            [self.up_field[0], self.up_field[1], self.up_field[2]]
        )

    @pos.setter
    def pos(self, v: np.ndarray):
        for i, vi in enumerate(v):
            self.pos_field[i] = vi

    @dir.setter
    def dir(self, v: np.ndarray):
        for i, vi in enumerate(v):
            self.dir_field[i] = vi

    @up.setter
    def up(self, v: np.ndarray):
        for i, vi in enumerate(v):
            self.up_field[i] = vi

    @ti.func
    def orthonormalize(self):
        up = self._up()
        dir = self._dir()
        x = up.cross(dir)
        up_perp = dir.cross(x)
        x = up_perp.cross(dir)
        return ti.math.mat3(x, up_perp, dir)

    @ti.func
    def init_ray(self, u: int, v: int, fov: float, res: ti.math.vec2, dcm: ti.math.mat3, is_perspective: bool):
        pos, d = self.init_ray_orthographic(u, v, fov=fov, res=res, dcm=dcm)
        if is_perspective:
            pos, d = self.init_ray_perspective(u, v, fov=fov, res=res, dcm=dcm)
        print(pos, d)
        return Ray(position=pos, direction=d, power=1.0)

    @ti.func
    def init_ray_orthographic(self, u: int, v: int, fov: float, res: ti.math.vec2, dcm: ti.math.mat3):
        aspect_ratio = res.x / res.y
        camera_x = dcm[0,:]
        camera_up_perp = dcm[1,:]

        frac_x = (v + 0) / res.y
        frac_y = (u + 0) / res.x
        pos = (
            self._pos()
            + fov * (frac_x - 0.5) * camera_up_perp
            + aspect_ratio * fov * (frac_y - 0.5) * camera_x
        )
        return pos, dcm[2,:]

    @ti.func
    def init_ray_perspective(self, u: int, v: int, fov: float, res: ti.math.vec2, dcm: ti.math.mat3):
        aspect_ratio = res.x / res.y
        pos = self._pos()
        d = (
            dcm
            @ ti.Vector(
                [
                    (
                        2 * fov * (u + 0) / res.y
                        - fov * aspect_ratio
                        - 1e-5
                    ),
                    2 * fov * (v + 0) / res.y - fov - 1e-5,
                    1.0,
                ]
            ).normalized()
        )
        return pos, d
