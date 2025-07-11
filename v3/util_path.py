import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax.random import PRNGKey, normal
import matplotlib.pyplot as plt
from functools import partial
from jax import vmap
from jax.tree import map
import time
import optax
from util import *
from tsit_tableau import *

(c1, c2, c3, c4, c5, c6, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76) = tsit_tableau()
coeffs = [
        -0.101321183346709072589001712988183609230944236760490476,  # x
        0.00662087952180793343258682906697112938547424931632185616,  # x^3
        -0.000173505057912483501491115906801116298084629719204655552,  # x^5
        2.52229235749396866288379170129828403876289663605034418e-6,  # x^7
    ]
@jax.jit
def sin_cheby(x):
    # Coefficients of the Chebyshev polynomial
    x2 = x * x  # x^2
    # Evaluate the polynomial
    p7 = coeffs[3]
    p5 = p7 * x2 + coeffs[2]
    p3 = p5 * x2 + coeffs[1]
    p1 = p3 * x2 + coeffs[0]
    fx = p1 * x

    dp7 = 7 * coeffs[3]
    dp5 = dp7 * x2 + 5 * coeffs[2]
    dp3 = dp5 * x2 + 3 * coeffs[1]
    dfx = dp3 * x2 + coeffs[0]

    # Approximate sin(x) using the Chebyshev polynomial
    sinx = (
        (x - 3.1415927410125732 + 0.00000008742277657347586) *
        (x + 3.1415927410125732 - 0.00000008742277657347586) *
        fx
    )

    diff_sinx = 2*x*fx - ((3.1415927410125732 - 0.00000008742277657347586)**2 - x2) * dfx
    return sinx, diff_sinx
@partial(jax.jit, static_argnums=(-3, -2, -1))
def rk4_odeint_path(input_, W_a, omega_a, t0, tf, dt):
    def func_(input_, t, W_a, omega_a):
        x, logp_x, d_logp_x, diff_xf_W, diff_xf_omega = input_
        t_k = W_a.shape[0] // 2

        W = W_t(W_a, t)
        a_w = jnp.concatenate(
            (jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        '''
        Omega t dependence
        omega = omega_t(omega_a, t)
        a_omega = jnp.concatenate(
            (jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        '''
        omega = omega_a
        sin_omegax = jnp.concatenate((omega[0]*x[None, :], jnp.sin(omega[1:].reshape(-1, 1, 1) * x)), axis=0)
        diff_sinx =  jnp.concatenate((omega[0]*jnp.ones_like(x)[None, :], omega[1:].reshape(-1, 1, 1) * jnp.cos(omega[1:].reshape(-1, 1, 1) * x)), axis=0)
        fft_sin_omegax = jnp.fft.fft2(sin_omegax)
        fft_W = jnp.fft.fft2(W)
        yf_fft = fft_W * fft_sin_omegax
        y = jnp.sum(jnp.fft.ifft2(yf_fft).real, axis=0)
        omega_cos_omegax = omega.reshape(-1, 1, 1) * diff_sinx
        # x_ = (omega.reshape(-1, 1, 1) * x)% (2 * jnp.pi) - jnp.pi
        # sin_omegax, diff_sinx = sin_cheby(x_)
        #print("omega_shape:", omega.reshape(-1, 1, 1).shape)
        #print("x:", x.shape)

        return (y,
        jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * (omega_cos_omegax)),
        -jnp.sum(omega.reshape(-1, 1, 1) * diff_sinx * jnp.fft.ifft2(
        jnp.flip(jnp.roll(fft_W, (-1, -1), (-1, -2)), (-1, -2)) * jnp.fft.fft2(d_logp_x)).real, axis=0) +
        jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * ((omega ** 2).reshape(-1, 1, 1) * sin_omegax), axis=0),
        a_w.reshape(-1, 1, 1, 1) * sin_omegax,
        jnp.fft.ifft2(fft_W * jnp.fft.fft2(omega.reshape(-1, 1, 1) * (x * diff_sinx))).real)
    '''
        Kinetic regularization
        # W: 2 * a_w.reshape(-1, 1, 1, 1) * jnp.fft.ifft2(yf_fft * jnp.fft.fft2(jnp.flip(jnp.roll(fft_sin_omegax, (-1, -1), axis=(-2, -1)), (-2, -1)))).real
        # omega: 2 * jnp.sum(y *  jnp.fft.ifft2(fft_W*jnp.fft.fft2(omega_cos_omegax)), axis = (-2, -1))
    '''

    """Integrate a system of ODEs using Tsit5."""

    def step_func(cur_y, cur_t, dt):
        k1 = func(cur_y, cur_t, W_a, omega_a)
        k2 = func(add_trees(cur_y, mul_const_tree(k1, dt * a21)), cur_t + dt * c1 , W_a, omega_a)
        k3 = func(add_trees(cur_y, mul_const_tree(k1, dt * a31), mul_const_tree(k2, dt * a32)),
                  cur_t + dt * c2, W_a, omega_a)
        k4 = func(add_trees(cur_y, mul_const_tree(k1, dt * a41), mul_const_tree(k2, dt * a42),
                            mul_const_tree(k3, dt * a43)), cur_t + dt * c3, W_a, omega_a)
        k5 = func(add_trees(cur_y, mul_const_tree(k1, dt * a51), mul_const_tree(k2, dt * a52),
                            mul_const_tree(k3, dt * a53), mul_const_tree(k4, dt * a54)), cur_t + dt * c4, W_a, omega_a)
        k6 = func(add_trees(cur_y, mul_const_tree(k1, dt * a61), mul_const_tree(k2, dt * a62),
                            mul_const_tree(k3, dt * a63), mul_const_tree(k4, dt * a64), mul_const_tree(k5, dt * a65)),
                  cur_t + dt, W_a, omega_a)
        final_step = add_trees(
            mul_const_tree(k1, dt * a71),
            mul_const_tree(k2, dt * a72),
            mul_const_tree(k3, dt * a73),
            mul_const_tree(k4, dt * a74),
            mul_const_tree(k5, dt * a75),
            mul_const_tree(k6, dt * a76)
        )
        return final_step

    def body_fun(y, t, dt):
        """Take one step of RK4."""
        dy = step_func(y, t, dt)
        return add_trees(y, dy), None

    func = vmap(func_, in_axes=(0, None, None, None))
    t = jnp.arange(t0, tf, dt)
    y1, t_dummy = jax.lax.scan(partial(body_fun, dt=dt), input_, t)
    return y1

