from typing import Tuple

import numpy as np
import taichi as ti

from .brdf import sample_ggx_micro_normal_world, ggx_reflectance, reflect
from .scenes import Scene
from .math import rdot, random_direction
from .camera import Camera, Ray


@ti.data_oriented
class RayMarchRenderer:
    def __init__(
        self,
        scene: Scene,
        camera: Camera,
        max_bounces: int,
        divergence_dist: float = 100.0,
        samples_per_pixel: int = 1,
        show_gui: bool = False,
        gui_fps_limit: int = 1_000,
        max_march_steps: int = 100,
    ) -> None:
        self.scene = scene
        self.camera = camera

        self.divergence_dist = divergence_dist
        self.max_bounces = max_bounces
        self.samples_per_pixel = samples_per_pixel
        self.max_march_steps = max_march_steps
        self.res = tuple([int(x) for x in self.camera.res])

        self.color_buffer = ti.Vector.field(1, dtype=ti.f32, shape=self.res)
        if show_gui:
            self.gui = ti.GUI("Mirari Ray Marcher", self.res)
            self.gui.fps_limit = gui_fps_limit

        self._j = 0

    def show(self):
        if not hasattr(self, "gui"):
            raise ValueError(
                "This RayMarchRenderer was initialized with show_gui=False, it has no gui to show"
            )
        img = self.color_buffer.to_numpy() / self.samples_per_pixel / self._j
        self.gui.set_image(img)
        self.gui.show()

    @ti.func
    def march(
        self,
        ray: Ray,
        divergence_dist: float,
        max_march_steps: int,
    ) -> Tuple[float, ti.types.struct]:
        j = 0
        dist_marched = 0.0
        closest_obj = self.scene.objects[0]
        while j < max_march_steps and dist_marched < divergence_dist:
            new_dist, closest_obj = self.scene.sdf(ray.position + dist_marched * ray.direction)
            dist_marched += new_dist
            if new_dist < 1e-6:
                break
            j += 1
        return [ti.min(divergence_dist, dist_marched), closest_obj]

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
        return img.sum() / self.samples_per_pixel

    def total_brightness(self):
        return self.sum() * self.camera.fov**2 / self.res[1] ** 2

    @ti.func
    def next_hit(self, ray: Ray, divergence_dist: float, max_march_steps: int):
        closest, normal = divergence_dist, ti.Vector.zero(ti.f32, 3)
        ray_march_dist, closest_obj = self.march(
            ray, divergence_dist, max_march_steps
        )
        if ray_march_dist < divergence_dist and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.sdf_normal(ray.position + ray.direction * closest)
        return closest, normal, closest_obj

    def reset_buffer(self):
        self._j = 0
        self._reset_buffer()

    @ti.kernel
    def _reset_buffer(self):
        for u, v in self.color_buffer:
            self.color_buffer[u, v] = 0.0

    def render(self, light_normal: ti.math.vec3):
        self._j += 1
        self._render(
            light_normal,
            self.samples_per_pixel,
            self.max_bounces,
            self.camera.fov,
            self.camera.res_vector,
            self.camera.is_perspective,
            self.divergence_dist,
            self.max_march_steps,
        )

    @ti.kernel
    def _render(
        self,
        light_normal: ti.math.vec3,
        samples_per_pixel: int,
        max_bounces: int,
        fov: float,
        res: ti.math.vec2,
        is_perspective: bool,
        divergence_dist: float,
        max_march_steps: int,
    ):
        dcm = self.camera.orthonormalize()

        for u, v in self.color_buffer:
            ti.loop_config(serialize=False)  # Serializes the next for loop
            for _ in range(samples_per_pixel):
                ray = self.camera.init_ray(u, v, fov=fov, res=res, dcm=dcm, is_perspective=is_perspective)
                ray = self.path_trace(
                    ray,
                    light_normal,
                    max_bounces=max_bounces,
                    divergence_dist=divergence_dist,
                    max_march_steps=max_march_steps,
                )
                self.color_buffer[u, v] += ray.power

    @ti.func
    def path_trace(
        self,
        ray: Ray,
        light_normal: ti.math.vec3,
        max_bounces: int,
        divergence_dist: float,
        max_march_steps: int,
    ):
        depth = 0
        last_surface_normal = light_normal

        ti.loop_config(serialize=False)
        while depth < max_bounces:
            closest, normal, closest_obj = self.next_hit(
                ray, divergence_dist, max_march_steps
            )
            depth += 1
            if depth == max_bounces:  # Then we hit no lights
                ray.power = 0
                break
            if closest == divergence_dist:  # Then we have diverged
                # power *= rdot(last_surface_normal, -light_normal)
                ray.power = 0
                break
            else:
                if closest_obj.material.emmissive:  # If we've hit a light
                    ray.power *= closest_obj.material.cs * rdot(-ray.direction, normal)
                    break
                last_surface_normal = normal
                hit_pos = ray.position + closest * ray.direction

                wo = -ray.direction

                wi = ti.math.vec3(0.0, 0.0, 0.0)
                if (
                    ti.random() < closest_obj.material.cs
                ):  # Then we've reflected specularly
                    wm = sample_ggx_micro_normal_world(
                        normal, closest_obj.material.a**2
                    )
                    wi = reflect(wo, wm)
                    refl = ggx_reflectance(
                        wi,
                        wo,
                        normal,
                        wm,
                        closest_obj.material.cs,
                        closest_obj.material.a**2,
                    )
                    ray.power *= refl
                else:  # Then we've reflected diffusely
                    wi = (normal + random_direction()).normalized()
                    ray.power *= rdot(wi, normal)

                dir = wi
                pos = hit_pos + 1e-5 * dir
                ray.position = pos
                ray.direction = dir
        if ti.math.isnan(ray.power):
            ray.power = 0.0
        return ray
