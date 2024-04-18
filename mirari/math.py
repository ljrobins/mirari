import numpy as np
import taichi as ti

@ti.func
def r1(t: float) -> ti.math.mat3:
    """Rotation about the second body axis with input in radians

    :param t: Angle t (short for theta) [rad]
    :type t: float
    :return: Rotation matrix(s) about the second body axis
    :rtype: np.ndarray
    """
    return ti.Matrix(
        [
            [1, 0, 0],
            [0, ti.cos(t), ti.sin(t)],
            [0, -ti.sin(t), ti.cos(t)],
        ]
    )

@ti.func
def r2(t: float) -> ti.math.mat3:
    """Rotation about the second body axis with input in radians

    :param t: Angle t (short for theta) [rad]
    :type t: float
    :return: Rotation matrix(s) about the second body axis
    :rtype: ti.math.mat3
    """
    return ti.Matrix(
        [
            [ti.cos(t), 0, -ti.sin(t)],
            [0, 1, 0],
            [ti.sin(t), 0, ti.cos(t)],
        ]
    )


@ti.func
def r3(t: float) -> ti.math.mat3:
    """Rotation about the third body axis with input in radians

    :param t: Angle t (short for theta) [rad]
    :type t: float
    :return: Rotation matrix(s) about the third body axis
    :rtype: ti.math.mat3
    """
    return ti.Matrix(
        [
            [ti.cos(t), ti.sin(t), 0],
            [-ti.sin(t), ti.cos(t), 0],
            [0, 0, 1],
        ]
    )

def r1_py(t: float) -> ti.math.mat3:
    """Rotation about the second body axis with input in radians

    :param t: Angle t (short for theta) [rad]
    :type t: float
    :return: Rotation matrix(s) about the second body axis
    :rtype: np.ndarray
    """
    return ti.Matrix(
        [
            [1, 0, 0],
            [0, ti.cos(t), ti.sin(t)],
            [0, -ti.sin(t), ti.cos(t)],
        ]
    )

def r2_py(t: float) -> ti.math.mat3:
    """Rotation about the second body axis with input in radians

    :param t: Angle t (short for theta) [rad]
    :type t: float
    :return: Rotation matrix(s) about the second body axis
    :rtype: ti.math.mat3
    """
    return ti.Matrix(
        [
            [ti.cos(t), 0, -ti.sin(t)],
            [0, 1, 0],
            [ti.sin(t), 0, ti.cos(t)],
        ]
    )


def r3_py(t: float) -> ti.math.mat3:
    """Rotation about the third body axis with input in radians

    :param t: Angle t (short for theta) [rad]
    :type t: float
    :return: Rotation matrix(s) about the third body axis
    :rtype: ti.math.mat3
    """
    return ti.Matrix(
        [
            [ti.cos(t), ti.sin(t), 0],
            [-ti.sin(t), ti.cos(t), 0],
            [0, 0, 1],
        ]
    )

@ti.func
def sph_to_cart(
    az: float, el: float
) -> ti.math.vec3:
    """Converts from spherical ``(azimuth, elevation, range)`` to unit Cartesian ``(x, y, z)``

    :param az: Azimuth [rad]
    :type az: float
    :param el: Elevation [rad]
    :type el: float
    :return: Cartesian ``(x, y, z)`` unit vector
    :rtype: ti.math.vec3
    """
    rcos_theta = ti.cos(el)
    x = rcos_theta * ti.cos(az)
    y = rcos_theta * ti.sin(az)
    z = np.sin(el)
    return ti.Vector([x, y, z])

@ti.func
def random_n1_p1() -> float:
    return 2*ti.random() - 1

@ti.func
def random_normal_value() -> float:
    theta = 2 * np.pi * ti.random()
    rho = ti.sqrt(-2 * ti.log(ti.random()))
    return rho * ti.cos(theta)


@ti.func
def random_direction() -> ti.math.vec3:
    return ti.Vector(
        [random_normal_value(), random_normal_value(), random_normal_value()]
    )


@ti.func
def random_hemisphere_direction(n) -> ti.math.vec3:
    eps = 1e-4
    u = ti.Vector([1.0, 0.0, 0.0])
    if abs(n[1]) < 1 - eps:
        u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    v = n.cross(u)
    phi = 2 * np.pi * ti.random()
    ay = ti.sqrt(ti.random())
    ax = ti.sqrt(1 - ay**2)
    return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n


@ti.func
def lerp(v1: float, v2: float, t: float) -> ti.math.vec3:
    return (t * v2 + (1 - t) * v1).normalized()

@ti.func
def slerp(n1: ti.math.vec3, n2: ti.math.vec3, t: float) -> ti.math.vec3:
    om = ti.acos(ti.math.dot(n1, n2))
    return (ti.sin((1-t)*om)*n1 + ti.sin(t*om)*n2)/ti.sin(om)

@ti.func
def rv_to_dcm(rv: ti.math.vec3) -> ti.math.mat3:
    theta = rv.norm()
    c = ti.cos(theta)
    s = ti.sin(theta)
    rv_hat = rv.normalized()
    n1 = rv_hat[0]
    n2 = rv_hat[1]
    n3 = rv_hat[2]

    return ti.Matrix([
        [c + n1**2*(1-c), n1*n2*(1-c) + n3*s, n1*n3*(1-c) - n2*s],
        [n2*n1*(1-c) - n3*s, c + n2**2*(1-c), n2*n3*(1-c) + n1*s],
        [n3*n1*(1-c) + n2*s, n3*n2*(1-c) - n1*s, c + n3**2*(1-c)]
        ])

