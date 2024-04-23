import time

import numpy as np

import taichi as ti
import mirari as mi
import matplotlib.pyplot as plt

def main():
    t_start = time.time()
    camera_pos = ti.Vector([0.0, 0.0, 4.0])
    camera_dir = -ti.Vector([0.0, 0, 1.0])
    camera_up = ti.Vector([0.0, 1.0, 0.0])
    light_normal = ti.Vector([0.0, 0.0, -1.0])

    last_t = 0
    i = 0
    interval = 1
    cam = mi.Camera(pos=camera_pos, 
                dir=camera_dir, 
                up=camera_up, 
                fov=0.4, 
                res=(400,400),
                is_perspective=True)
    tracer = mi.RayMarchRenderer(scene=mi.Scene(objects=mi.simple_scene()), 
                                 camera=cam,
                                 max_depth=5,
                                 samples_per_pixel=1,
                                 show_gui=True)
    

    m = ti.aot.Module(ti.metal)
    m.add_kernel(tracer.render)

    m.save('test.tcm')
    endd

    totals = []
    while tracer.gui.running:
        tracer.render_image(light_normal)
        # cam.pos = cam.pos + 0.01
        if i % interval == 0:
            print(f"{interval / (time.time() - last_t):.2f} samples/s")
            last_t = time.time()
            tracer.show()
            totals.append(tracer.total_brightness())
        i += 1
        # tracer.reset_buffer()
        # if i == 1000:
        #     break


if __name__ == "__main__":
    main()