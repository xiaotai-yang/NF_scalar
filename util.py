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
from tsit_tableau import *

from tsit_tableau import *

(c1, c2, c3, c4, c5, c6, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76) = tsit_tableau()
@jax.jit
def initial_mask(A, L):
    A += jnp.transpose(A, (1, 0))
    A += A[::-1, :]
    A += A[:, ::-1]
    A/= 8
    return A[:-1, :-1]
@jax.jit
def backward_propagate_grads(B, L):
    # Step 1: Pad the 4x4 gradients to match the 5x5 matrix by adding zeros
    A = jnp.pad(B, ((0, 1), (0, 1)), mode='constant')

    # Step 2: Reverse the rolling (adjust this step if rolling was applied in your forward function)

    # Step 3: Undo the averaging by distributing the gradient contributions
    # Since the averaging involved adding the matrix to itself in various transformations,
    # we need to create a full 5x5 matrix that sums all these transformations' contributions
    A+= jnp.transpose(A, (1, 0))
    A+= A[::-1, :]
    A+= A[:, ::-1]
    A = A/8
    return A

@jax.jit
def phi4_action(phi, m2, lam):
    """Compute the Euclidean action for the scalar phi^4 theory.

    The Lagrangian density is kin(phi) + m2 * phi + l * phi^4

    Args:
        phi: Single field configuration of shape L^d.
        m2: Mass squared term (can be negative).
        lam: Coupling constant for phi^4 term.

    Returns:
        Scalar, the action of the field configuration..
    """

    a = jnp.sum(m2 * phi ** 2)
    if lam is not None:
        a += jnp.sum(lam * phi ** 4)
    # Kinetic term
    a += 2*jnp.sum(jnp.array([phi*(phi - jnp.roll(phi, 1, d)/2 - jnp.roll(phi, -1, d)/2)  for d in range(len(phi.shape))]))

    return a

@jax.jit
def diff_phi4_action(phi, m2, lam):
    a = 2 * m2 * phi
    if lam is not None:
        a += 4 * lam * phi ** 3
    for d in range(len(phi.shape)):
        a += 2*(2*phi - jnp.roll(phi, 1, d) - jnp.roll(phi, -1, d))
    return a



def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2 * jax.scipy.special.logsumexp(logw, axis=0) - jax.scipy.special.logsumexp(2 * logw, axis=0)
    ess_per_cfg = jnp.exp(log_ess) / len(logw)
    return ess_per_cfg


def normal_pdf(x):
    """Calculate the PDF of a standard normal distribution."""
    return (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * x ** 2)


def get_batch(num_samples, L, seed):
    #   Generate a batch of samples from a standard normal distribution.
    key = PRNGKey(seed+1)
    x = normal(key, (num_samples, L, L))
    logp_x = jnp.sum(jnp.log(normal_pdf(x)), axis=(1, 2))  # log probability of each sample

    return (x, logp_x, -x)


# @partial(jax.custom_vjp, nondiff_argnums=(0, 1))

def W_t(a, t):
    b = a.shape[0]//2
    return a[0]+jnp.sum(a[1:b+1]*(jnp.sin((jnp.arange(b)+1)*t)).reshape(-1, 1, 1, 1), axis = 0) + jnp.sum(a[b+1:]*(jnp.cos((jnp.arange(b)+1)*t)).reshape(-1, 1, 1, 1), axis = 0)
def omega_t(a, t):
    b = a.shape[0]//2
    return a[0]+jnp.sum(a[1:b+1]*(jnp.sin((jnp.arange(b)+1)*t)).reshape(-1, 1), axis = 0) + jnp.sum(a[b+1:]*(jnp.cos((jnp.arange(b)+1)*t)).reshape(-1, 1), axis = 0)

def mul_const_tree(tree, const):
    """Multiplies every element in the pytree by a constant."""
    return jax.tree.map(lambda x: x * const, tree)


def add_trees(*trees):
    """Adds multiple pytrees together."""
    return jax.tree.map(lambda *xs: sum(xs), *trees)


@jax.jit
def rk4_odeint(step_size, input_, ts, W_a, omega_a):
    def func_(input_, t, W_a, omega_a):
        x, logp_x, d_logp_x, diff_xf_W, diff_xf_omega = input_
        # jax.debug.print("W_a:{}", W_a.shape)

        W = W_t(W_a, t)
        omega = omega_t(omega_a, t)
        t_k = W_a.shape[0]//2
        a_w = jnp.concatenate((jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        a_omega = jnp.concatenate((jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        return (
        jnp.sum(jnp.fft.ifft2(jnp.fft.fft2(W) * jnp.fft.fft2(jnp.sin(omega.reshape(-1, 1, 1) * x))).real, axis = 0),
        jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * (omega.reshape(-1, 1, 1) * jnp.cos(omega.reshape(-1, 1, 1) * x))),
        -jnp.sum(omega.reshape(-1, 1, 1) * jnp.cos(omega.reshape(-1, 1, 1) * x) * jnp.fft.ifft2(
        jnp.flip(jnp.roll(jnp.fft.fft2(W), (-1, -1), (-1, -2)), (-1, -2)) * jnp.fft.fft2(d_logp_x)).real,
        axis=0) + jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * ((omega ** 2).reshape(-1, 1, 1) * jnp.sin(omega.reshape(-1, 1, 1) * x)), axis=0),
        a_w.reshape(-1, 1, 1, 1) * jnp.sin(omega.reshape(-1, 1, 1) * x),
        jnp.fft.ifft2(jnp.fft.fft2(W)*jnp.fft.fft2(a_omega.reshape(-1, 1, 1, 1)*(x*jnp.cos(omega.reshape(-1, 1, 1)*x)))).real)

    """Integrate a system of ODEs using the 4th order Runge-Kutta method."""

    def step_func(cur_y, cur_t, dt):
        """Take one step of RK4."""
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
                  cur_t + dt , W_a, omega_a)
        final_step = add_trees(
            mul_const_tree(k1, dt * a71),
            mul_const_tree(k2, dt * a72),
            mul_const_tree(k3, dt * a73),
            mul_const_tree(k4, dt * a74),
            mul_const_tree(k5, dt * a75),
            mul_const_tree(k6, dt * a76)
        )
        return final_step

    def cond_fun(carry):
        """Check if we've reached the last timepoint."""
        cur_y, cur_t = carry
        return cur_t < ts[1]

    def body_fun(carry):
        """Take one step of RK4."""
        cur_y, cur_t = carry
        next_t = jnp.minimum(cur_t + step_size, ts[1])
        dt = next_t - cur_t
        dy = step_func(cur_y, cur_t, dt)
        return add_trees(cur_y, dy), next_t

    func = vmap(func_, in_axes=(0, None, None, None))
    init_carry = (input_, ts[0])
    y1, t1 = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return y1



