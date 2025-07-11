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
@partial(jax.jit, static_argnums=(-3, -2, -1))
def rk4_odeint_path(input_, W_a, omega_a, t0, tf, dt):
    def func_(input_, t, W_a, omega_a):
        x, logp_x, d_logp_x, diff_xf_W, diff_xf_omega = input_
        W = W_t(W_a, t)
        omega = omega_t(omega_a, t)
        t_k = W_a.shape[0] // 2

        a_w = jnp.concatenate(
            (jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        a_omega = jnp.concatenate(
            (jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        return (
        jnp.sum(jnp.fft.ifft2(jnp.fft.fft2(W) * jnp.fft.fft2(jnp.sin(omega.reshape(-1, 1, 1) * x))).real, axis=0),
        jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * (omega.reshape(-1, 1, 1) * jnp.cos(omega.reshape(-1, 1, 1) * x))),
        -jnp.sum(omega.reshape(-1, 1, 1) * jnp.cos(omega.reshape(-1, 1, 1) * x) * jnp.fft.ifft2(
            jnp.flip(jnp.roll(jnp.fft.fft2(W), (-1, -1), (-1, -2)), (-1, -2)) * jnp.fft.fft2(d_logp_x)).real,
                 axis=0) + jnp.sum(
            W[:, 0, 0].reshape(-1, 1, 1) * ((omega ** 2).reshape(-1, 1, 1) * jnp.sin(omega.reshape(-1, 1, 1) * x)),
            axis=0),
        a_w.reshape(-1, 1, 1, 1) * jnp.sin(omega.reshape(-1, 1, 1) * x),
        jnp.fft.ifft2(jnp.fft.fft2(W) * jnp.fft.fft2(
            a_omega.reshape(-1, 1, 1, 1) * (x * jnp.cos(omega.reshape(-1, 1, 1) * x)))).real)

    """Integrate a system of ODEs using the 4th order Runge-Kutta method."""

    def step_func(cur_y, cur_t, dt):
        """Take one step of RK4."""
        '''
        Usual RK4
        k1 = func(cur_y, cur_t, W_a, omega_a)
        k2 = func(add_trees(cur_y, mul_const_tree(k1, dt * 0.4)), cur_t + dt * 0.4, W_a, omega_a)
        k3 = func(add_trees(cur_y, mul_const_tree(k1, dt * 0.29697761), mul_const_tree(k2, dt * 0.15875964)),
                  cur_t + dt * 0.45573725, W_a, omega_a)
        k4 = func(add_trees(cur_y, mul_const_tree(k1, dt * 0.21810040), mul_const_tree(k2, -dt * 3.05096516),
                            mul_const_tree(k3, dt * 3.83286476)), cur_t + dt, W_a, omega_a)

        final_step = add_trees(
            mul_const_tree(k1, dt * 0.17476028),
            mul_const_tree(k2, -dt * 0.55148066),
            mul_const_tree(k3, dt * 1.20553560),
            mul_const_tree(k4, dt * 0.17118478)
        )
        '''
        k1 = func(cur_y, cur_t, W_a, omega_a)
        k2 = func(add_trees(cur_y, mul_const_tree(k1, dt * a21)), cur_t + c1 * 0.4, W_a, omega_a)
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

