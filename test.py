import time

import numpy as np

import taichi as ti
import mirari as mi
import matplotlib.pyplot as plt

# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm


def main():
    t_start = time.time()
    camera_pos = ti.Vector([0, 1, 3])
    camera_dir = -camera_pos.normalized()
    camera_up = ti.Vector([0.0, 1.0, 0.0]).normalized()
    light_normal = ti.Vector([0.0, -1.0, 0.0]).normalized()
    last_t = 0
    i = 0
    interval = 1
    totals = []
    tracer = mi.RayMarchRenderer(scene=mi.Scene(objects=mi.scene_three_objs()), 
                                 res=(400,400),
                                 max_depth=4, 
                                 samples_per_pixel=10,
                                 show_gui=True)


    while tracer.gui.running:
        rotm = mi.r2(0.01)
        camera_pos = rotm @ camera_pos
        camera_dir = -camera_pos.normalized()
        cam = mi.Camera(pos=camera_pos, dir=camera_dir, up=camera_up, 
                                    fov=0.3, 
                                    res=ti.Vector(tracer.res),
                                    is_perspective=True)
        cam2 = mi.Camera(pos=camera_pos, dir=camera_dir, up=camera_up, 
                                    fov=2.0, 
                                    res=ti.Vector(tracer.res),
                                    is_perspective=False)
        # light_normal = rotm @ light_normal
        tracer.render_image(light_normal, cam)
        if i % interval == 0:
            print(f"{interval / (time.time() - last_t):.2f} samples/s")
            last_t = time.time()
            
            totals.append(tracer.total_brightness(cam))
            tracer.show()
        i += 1
        tracer.reset_buffer()
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
    # plt.show()
    # print(totals[-1])


if __name__ == "__main__":
    main()