

import taichi as ti
import numpy as np

@ti.data_oriented
class Camera:

    def __init__(self, pos, dir, up, res, fov, is_perspective: bool):
        self.pos_field = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
        self.dir_field = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
        self.up_field = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
        self.res = res
        self.fov = fov
        self.is_perspective = is_perspective

        for i,(pi,di,ui) in enumerate(zip(pos, dir, up)):
            self.pos_field[0][i] = pi
            self.dir_field[0][i] = di
            self.up_field[0][i] = ui
        
        # self.pos = ti.Vector([self.pos_field[0][0], self.pos_field[0][1], self.pos_field[0][2]])
        # self.dir = ti.Vector([self.dir_field[0][0], self.dir_field[0][1], self.dir_field[0][2]])
        # self.up = ti.Vector([self.up_field[0][0], self.up_field[0][1], self.up_field[0][2]])

    
    @ti.func
    def _pos(self):
        return ti.Vector([self.pos_field[0][0], self.pos_field[0][1], self.pos_field[0][2]])
    
    @ti.func
    def _dir(self):
        return ti.Vector([self.dir_field[0][0], self.dir_field[0][1], self.dir_field[0][2]])
    
    @ti.func
    def _up(self):
        return ti.Vector([self.up_field[0][0], self.up_field[0][1], self.up_field[0][2]])
    
    @property
    def pos(self):
        return ti.Vector([self.pos_field[0][0], self.pos_field[0][1], self.pos_field[0][2]])

    @property
    def dir(self):
        return ti.Vector([self.dir_field[0][0], self.dir_field[0][1], self.dir_field[0][2]])

    @property
    def up(self):
        return ti.Vector([self.up_field[0][0], self.up_field[0][1], self.up_field[0][2]])
    
    @pos.setter
    def pos(self, v: np.ndarray):
        for i,vi in enumerate(v):
            self.pos_field[0][i] = vi

    @dir.setter
    def pos(self, v: np.ndarray):
        for i,vi in enumerate(v):
            self.dir_field[0][i] = vi

    @up.setter
    def pos(self, v: np.ndarray):
        for i,vi in enumerate(v):
            self.up_field[0][i] = vi


    @ti.func
    def aspect_ratio(self):
        return self.res[0] / self.res[1]
    
    @ti.func
    def orthonormalize(self):
        up = self._up()
        dir = self._dir()
        x = ti.math.cross(up, dir)
        up_perp = ti.math.cross(dir, x)
        x = ti.math.cross(up_perp, dir)
        return x, up_perp, dir
    
    @ti.func
    def init_ray(self, u: int, v: int):
        pos, d = self.init_ray_orthographic(u, v)
        if self.is_perspective:
            pos, d = self.init_ray_perspective(u, v)
        return pos, d
    
    @ti.func
    def init_ray_orthographic(self, u: int, v: int):
        camera_x, camera_up_perp, _ = self.orthonormalize()
        d = self._dir()
        frac_x = (v+ti.random()) / self.res[1]
        frac_y = (u+ti.random()) / self.res[0]
        pos = self._pos() \
            + self.fov * (frac_x-0.5) * camera_up_perp \
            + self.aspect_ratio() * self.fov * (frac_y-0.5) * camera_x
        return pos, d

    @ti.func
    def init_ray_perspective(self, u: int, v: int):
        camera_x, camera_up_perp, _ = self.orthonormalize()
        dir = self._dir()
        camera_dcm = ti.Matrix([[camera_x[0], camera_x[1], camera_x[2]], 
                                [camera_up_perp[0], camera_up_perp[1], camera_up_perp[2]],
                                [dir[0], dir[1], dir[2]]])
        pos = self._pos()
        d = camera_dcm.transpose() @ ti.Vector(
            [
                (
                    2 * self.fov * (u + ti.random()) / self.res[1]
                    - self.fov * self.aspect_ratio()
                    - 1e-5
                ),
                2 * self.fov * (v + ti.random()) / self.res[1] - self.fov - 1e-5,
                1.0,
            ]
        ).normalized()
        return pos, d
