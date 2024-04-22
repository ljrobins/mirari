

import taichi as ti

@ti.dataclass
class Camera:
    pos: ti.math.vec3
    dir: ti.math.vec3
    up: ti.math.vec3
    res: ti.math.vec2
    fov: float
    is_perspective: bool

    @ti.func
    def aspect_ratio(self):
        return self.res[0] / self.res[1]
    
    @ti.func
    def orthonormalize(self):
        x = ti.math.cross(self.up, self.dir)
        up_perp = ti.math.cross(self.dir, x)
        x = ti.math.cross(up_perp, self.dir)
        return x, up_perp, self.dir
    
    @ti.func
    def init_ray(self, u: int, v: int):
        pos, d = self.init_ray_orthographic(u, v)
        if self.is_perspective:
            pos, d = self.init_ray_perspective(u, v)
        return pos, d
    
    @ti.func
    def init_ray_orthographic(self, u: int, v: int):
        camera_x, camera_up_perp, _ = self.orthonormalize()
        d = self.dir
        frac_x = (v+ti.random()) / self.res[1]
        frac_y = (u+ti.random()) / self.res[0]
        pos = self.pos \
            + self.fov * (frac_x-0.5) * camera_up_perp \
            + self.aspect_ratio() * self.fov * (frac_y-0.5) * camera_x
        return pos, d

    @ti.func
    def init_ray_perspective(self, u: int, v: int):
        camera_x, camera_up_perp, _ = self.orthonormalize()
        camera_dcm = ti.Matrix([[camera_x[0], camera_x[1], camera_x[2]], 
                                [camera_up_perp[0], camera_up_perp[1], camera_up_perp[2]],
                                [self.dir[0], self.dir[1], self.dir[2]]])
        pos = self.pos
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
