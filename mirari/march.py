from typing import Tuple

import numpy as np
import taichi as ti

from .brdf import sample_ggx_micro_normal_world, ggx_reflectance, reflect
from .scenes import Scene
from .math import rdot


a2 = 0.01**2
cs = 0.95


@ti.data_oriented
class RayMarchRenderer:
    def __init__(
        self,
        scene: Scene,
        fov: float,
        res: Tuple[float, float],
        max_depth: int,
        divergence_dist: float = 100.0,
        samples_per_pixel: int = 1,
        show_gui: bool = False,
        gui_fps_limit: int = 1_000,
        max_march_steps: int = 100,
    ) -> None:
        self.scene = scene

        self.fov = fov
        self.inf = divergence_dist
        self.res = res
        self.max_depth = max_depth
        self.aspect_ratio = self.res[0] / self.res[1]
        self.samples_per_pixel = samples_per_pixel
        self.max_march_steps = max_march_steps

        self.color_buffer = ti.Vector.field(1, dtype=ti.f32, shape=self.res)
        if show_gui:
            self.gui = ti.GUI("Mirari Ray Marcher", res)
            self.gui.fps_limit = gui_fps_limit
        
        self._j = 0

    def show(self):
        if not hasattr(self, "gui"):
            raise ValueError(
                "This RayMarchRenderer was initialized with show_gui=False, it has no gui to show"
            )
        img = self.color_buffer.to_numpy() / self._j
        self.gui.set_image(img)
        self.gui.show()

    @ti.func
    def march(self, p: ti.math.vec3, d: ti.math.vec3) -> Tuple[float, ti.types.struct]:
        j = 0
        dist_marched = 0.0
        closest_obj = self.scene.objects[0]
        while j < self.max_march_steps and dist_marched < self.inf:
            new_dist, closest_obj = self.scene.sdf(p + dist_marched * d)
            dist_marched += new_dist
            if new_dist < 1e-6:
                break
            j += 1
        return [ti.min(self.inf, dist_marched), closest_obj]

    @ti.func
    def sdf_normal(self, p):
        d = 1e-3
        n = ti.Vector([0.0, 0.0, 0.0])
        sdf_center, _ = self.scene.sdf(p)
        for i in ti.static(range(3)):
            inc = p
            inc[i] += d
            n[i] = (1 / d) * (self.scene.sdf(inc)[0] - sdf_center)
        return n.normalized()

    def sum(self):
        img = self.color_buffer.to_numpy()
        if np.any(np.isnan(img)):
            raise ValueError("A pixel in the image is nan, aborting!")
        return img.sum() / self._j

    def total_brightness(self):
        return self.sum() * self.fov**2 / self.res[1] ** 2

    @ti.func
    def next_hit(self, pos, d):
        """Computes the next intersection point

        :param pos: _description_
        :type pos: _type_
        :param d: _description_
        :type d: _type_
        :return: _description_
        :rtype: _type_
        """
        closest, normal = self.inf, ti.Vector.zero(ti.f32, 3)
        ray_march_dist, closest_obj = self.march(pos, d)
        if ray_march_dist < self.inf and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.sdf_normal(pos + d * closest)
        return closest, normal, closest_obj

    def reset_buffer(self):
        self._reset_buffer()
        self._j = 0

    @ti.kernel
    def _reset_buffer(self):
        for u, v in self.color_buffer:
            self.color_buffer[u, v] = 0.0

    def render_image(
        self,
        camera_pos: ti.math.vec3,
        camera_dir: ti.math.vec3,
        camera_up: ti.math.vec3,
        light_normal: ti.math.vec3,
    ):
        for _ in range(self.samples_per_pixel):
            self.render(camera_pos, camera_dir, camera_up, light_normal)
            self._j += 1


    @ti.kernel
    def render(self, 
               camera_pos: ti.math.vec3, 
               camera_dir: ti.math.vec3,
               camera_up: ti.math.vec3,
               light_normal: ti.math.vec3):
        camera_x = ti.math.cross(camera_up, camera_dir)
        camera_up_perp = ti.math.cross(camera_dir, camera_x)
        camera_x = ti.math.cross(camera_up_perp, camera_dir)
        # camera_dcm = ti.Matrix([[camera_x[0], camera_x[1], camera_x[2]], 
        #                         [camera_up[0], camera_up[1], camera_up[2]],
        #                         [camera_dir[0], camera_dir[1], camera_dir[2]]])
        # d = ti.Vector(
        #     [
        #         (
        #             2 * self.fov * (u + ti.random()) / self.res[1]
        #             - self.fov * self.aspect_ratio
        #             - 1e-5
        #         ),
        #         2 * self.fov * (v + ti.random()) / self.res[1] - self.fov - 1e-5,
        #         -1.0,
        #     ]
        # ).normalized()
        for u, v in self.color_buffer:
            d = camera_dir
            frac_x = (v+ti.random()) / self.res[1]
            frac_y = (u+ti.random()) / self.res[0]
            pos = camera_pos \
                + self.fov * (frac_x-0.5) * camera_up \
                + self.aspect_ratio * self.fov * (frac_y-0.5) * camera_x

            depth = 0
            power = 1.0
            last_surface_normal = light_normal

            while depth < self.max_depth:
                closest, normal, closest_obj = self.next_hit(pos, d)
                depth += 1
                if closest == self.inf:
                    power *= rdot(last_surface_normal, -light_normal)
                    break
                else:
                    last_surface_normal = normal
                    hit_pos = pos + closest * d

                    wo = -d

                    wm = sample_ggx_micro_normal_world(normal, closest_obj.material.a**2)
                    wi = reflect(wo, wm)
                    refl = ggx_reflectance(wi, wo, normal, wm, closest_obj.material.cs, closest_obj.material.a**2)
                    power *= refl

                    d = wi
                    pos = hit_pos + 1e-5 * d


                    
            self.color_buffer[u, v] += power if not ti.math.isnan(power) else 0
            
