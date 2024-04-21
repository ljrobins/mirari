import taichi as ti
import numpy as np

from .math import random_direction, rdot, sph_to_cart, reflect


@ti.func
def ggx(M: ti.math.vec3, N: ti.math.vec3, a2: float) -> float:
    """Calculates the GGX normal distribution function

    :param M: Microfacet surface normal direction
    :type M: ti.math.vec3
    :param N: Macro surface normal
    :type N: ti.math.vec3
    :param a2: Surface roughness, squared
    :type a2: float
    :return: Normal distribution density
    :rtype: float
    """
    return a2 / (np.pi * (rdot(N, M) ** 2 * (a2 - 1) + 1) ** 2)

@ti.func
def g_smith(
    O: ti.math.vec3, N: ti.math.vec3, L: ti.math.vec3, a2: float
) -> float:
    # https://schuttejoe.github.io/post/ggximportancesamplingpart1/
    ndl = rdot(N, L)
    ndo = rdot(N, O)
    num = 2 * ndl * ndo
    den1 = ndo * ti.sqrt(a2 + (1 - a2) * ndl**2)
    den2 = ndl * ti.sqrt(a2 + (1 - a2) * ndo**2)
    return num / (den1 + den2)

@ti.func
def fresnel_schlick(
    H: ti.math.vec3, L: ti.math.vec3, cs: float
) -> float:
    return cs + (1 - cs) * (1 - rdot(H, L)) ** 5

@ti.func
def sample_ggx_micro_normal_tangent(a2: float) -> ti.math.vec3:
    e1 = ti.random()
    e2 = ti.random()

    # theta = ti.atan2(ti.sqrt(a2*e1/(1-e1)),1) # GGX
    theta = ti.acos(ti.sqrt((1-e1)/(e1*(a2-1)+1))) # GGX
    phi = 2 * np.pi * e2
    normal_tangent = sph_to_cart(phi, np.pi/2-theta)
    return normal_tangent

@ti.func
def sample_ggx_micro_normal_world(N: ti.math.vec3, a2: float) -> ti.math.vec3:
    normal_tangent = sample_ggx_micro_normal_tangent(a2)
    x = ti.math.cross(random_direction(), N).normalized()
    y = ti.math.cross(N, x)
    dcm = ti.Matrix([[x[0], y[0], N[0]], 
                     [x[1], y[1], N[1]], 
                     [x[2], y[2], N[2]]])
    wm = dcm @ normal_tangent # The micro normal vector
    return wm

@ti.func
def ggx_reflectance(wi: ti.math.vec3, 
                    wo: ti.math.vec3, 
                    N: ti.math.vec3, 
                    wm: ti.math.vec3, 
                    cs: float,
                    a2: float):
    F = fresnel_schlick(wm, wi, cs)
    G2 = g_smith(wi, N, wo, ti.sqrt(a2))
    weight = ti.abs(ti.math.dot(wo, wm)) / (ti.math.dot(wo, N) * ti.math.dot(wm, N))
    integrand_importance = F * G2 * weight
    ok = (ti.math.dot(N, wi) > 0.0) & (ti.math.dot(wi, wm) > 0.0)
    if not ok:
        integrand_importance = 0
    return integrand_importance
