import time

import numpy as np

import taichi as ti
import mirari as mi
import matplotlib.pyplot as plt

# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm


def main():
    t_start = time.time()
    camera_pos = 10*ti.Vector([0, 1, 3])
    camera_dir = -camera_pos.normalized()
    camera_up = ti.Vector([0.0, 1.0, 0.0]).normalized()
    light_normal = ti.Vector([0.0, -1.0, 0.0]).normalized()
    last_t = 0
    i = 0
    interval = 5
    totals = []
    tracer = mi.RayMarchRenderer(fov=2.0, 
                                 scene=mi.Scene(sdf=mi.scene_one), 
                                 res=(740, 480), 
                                 max_depth=8, 
                                 samples_per_pixel=1,
                                 show_gui=True)

    rotm = mi.r3(0.01*ti.cos(25/50.0))
    # camera_pos = rotm @ camera_pos
    # camera_dir = rotm @ camera_dir
    light_normal = rotm @ light_normal

    while tracer.gui.running:
        tracer.render_image(camera_pos, camera_dir, camera_up, light_normal)
        if i % interval == 0:
            print(f"{interval / (time.time() - last_t):.2f} samples/s")
            last_t = time.time()
            
            totals.append(tracer.total_brightness())
            tracer.show()
        i += 1
        # tracer.reset_buffer()
        # if i == 1000:
        #     break
    
    print(time.time()-t_start)

    totals = np.array(totals)
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    percent_error = (totals - totals[-1]) / totals[-1] * 100
    plt.plot(totals)
    plt.title("Image sum over time")
    plt.xlabel("Iteration")
    plt.ylabel("Image sum")
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(percent_error)
    plt.xlabel("Iteration")
    plt.ylabel("Percent error")
    plt.grid()
    plt.tight_layout()
    plt.show()
    # print(totals[-1])


if __name__ == "__main__":
    main()