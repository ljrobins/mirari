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
                res=(499,500),
                is_perspective=True)
    
    tracer = mi.RayMarchRenderer(scene=mi.Scene(objects=mi.cornell_box_scene()), 
                                 camera=cam,
                                 max_bounces=4,
                                 samples_per_pixel=1,
                                 show_gui=True,
                                 divergence_dist=1e2)

    totals = []
    while True:
        last_t = time.time()
        tracer.render(light_normal)
        totals.append(tracer.total_brightness())
        print(f"{interval / (time.time() - last_t):.2f} samples/s")

        if i % interval == 0:
            tracer.show()
        i += 1
        # tracer.reset_buffer()
        if i == 100:
            break
    
    ti.profiler.print_scoped_profiler_info()

    arr = np.flipud(tracer.color_buffer.to_numpy().swapaxes(0,1))
    plt.imsave("test.png", arr=np.tile(np.clip(arr, 0, 255)/255, (1,1,3)))

    # totals = np.array(totals)
    # plt.figure(figsize=(8,4))
    # plt.subplot(1,2,1)
    # percent_error = (totals - totals[-1]) / totals[-1] * 100
    # plt.plot(totals)
    # plt.title("Image sum over time")
    # plt.xlabel("Iteration")
    # plt.ylabel("Image sum")
    # plt.grid()
    # plt.subplot(1,2,2)
    # plt.plot(percent_error)
    # plt.xlabel("Iteration")
    # plt.ylabel("Percent error")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()