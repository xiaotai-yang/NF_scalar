import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax.random import PRNGKey, normal, multivariate_normal
import matplotlib.pyplot as plt
from functools import partial
from jax import vmap
from jax.tree import map
import time
import optax
from tsit_tableau import *
(c1, c2, c3, c4, c5, c6, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76) = tsit_tableau()
batch_dot1 = jax.jit(vmap(jnp.dot, (0, 0), 0))
batch_dot2 = jax.jit(vmap(jnp.dot, (None, 0), 0))
def initial_mask(A):
    A += jnp.transpose(A, (1, 0))
    A += A[::-1, :]
    A += A[:, ::-1]
    A /= 8
    return A[:-1, :-1]

def backward_propagate_grads(B):
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
    a += jnp.sum(jnp.array([phi*(phi - jnp.roll(phi, 1, d)/2 - jnp.roll(phi, -1, d)/2)  for d in range(len(phi.shape))]))

    return a

@jax.jit
def diff_phi4_action(phi, m2, lam):
    a = 2 * m2 * phi
    if lam is not None:
        a += 4 * lam * phi ** 3
    for d in range(len(phi.shape)):
        a += 2*phi - jnp.roll(phi, 1, d) - jnp.roll(phi, -1, d)
    return a

def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2 * jax.scipy.special.logsumexp(logw, axis=0) - jax.scipy.special.logsumexp(2 * logw, axis=0)
    ess_per_cfg = jnp.exp(log_ess) / len(logw)
    return ess_per_cfg

def normal_pdf(x):
    """Calculate the PDF of a standard normal distribution."""
    return (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * x ** 2)

def construct_covariance(nx, ny, m2):
    """
    Constructs the discrete Laplacian matrix for a 2D nx x ny lattice with periodic boundary conditions.

    Parameters:
    - nx, ny: Number of lattice points in the x and y directions.

    Returns:
    - L: The Laplacian matrix of size (nx*ny) x (nx*ny).
    """
    N = nx * ny
    La = jnp.eye(N)*(4 - 2 * m2) * 2

    for x in range(nx):
        for y in range(ny):
            index = x * ny + y
            # Periodic boundary conditions: neighbors wrap around
            neighbors = [
                ((x + 1) % nx, y),  # Right
                ((x - 1) % nx, y),  # Left
                (x, (y + 1) % ny),  # Up
                (x, (y - 1) % ny)  # Down
            ]

            # Off-diagonal elements
            for (xn, yn) in neighbors:
                neighbor_index = xn * ny + yn
                La = La.at[index, neighbor_index].set(-2.0)
    cov = jnp.linalg.inv(La)
    logdetla = jnp.log(jnp.linalg.det(La))
    C = jnp.linalg.cholesky(cov)
    return  La, cov, logdetla,  C

def get_batch(num_samples, L, La, logdetla, C, seed, m2, lam):
    #   Generate a batch of samples from a standard normal distribution.
    key = PRNGKey(seed)
    x0 = normal(key, shape = (num_samples, L**2))
    x = batch_dot2(C, x0)
    logp_x = -0.5*jnp.sum(x0**2, axis = 1)-0.5*logdetla-0.5*L**2*jnp.log(2*jnp.pi)-jnp.log(2)
    diff_logp_x = -batch_dot2(La, x).reshape(num_samples, L, L)
    phi0 = jnp.tile(jnp.array([[jnp.sqrt(-m2/(2*lam)), -jnp.sqrt(-m2/(2*lam))]]), (num_samples, L**2//2))
    x = (x+phi0).reshape(num_samples, L, L)
    #logp_x = jnp.sum(jnp.log(normal_pdf(x)), axis=(1, 2))  # log probability of each sample

    return (x, logp_x, diff_logp_x)

# @partial(jax.custom_vjp, nondiff_argnums=(0, 1))

def W_t(a, t):
    b = a.shape[0] // 2
    return a[0] + jnp.sum(a[1:b + 1] * (jnp.sin((jnp.arange(b) + 1) * t)).reshape(-1, 1, 1, 1), axis=0) + jnp.sum(
        a[b + 1:] * (jnp.cos((jnp.arange(b) + 1) * t)).reshape(-1, 1, 1, 1), axis=0)


def omega_t(a, t):
    b = a.shape[0] // 2
    return a[0] + jnp.sum(a[1:b + 1] * (jnp.sin((jnp.arange(b) + 1) * t)).reshape(-1, 1), axis=0) + jnp.sum(
        a[b + 1:] * (jnp.cos((jnp.arange(b) + 1) * t)).reshape(-1, 1), axis=0)


def mul_const_tree(tree, const):
    """Multiplies every element in the pytree by a constant."""
    return jax.tree.map(lambda x: x * const, tree)


def add_trees(*trees):
    """Adds multiple pytrees together."""
    return jax.tree.map(lambda *xs: sum(xs), *trees)


@partial(jax.jit, static_argnums=(-3, -2, -1))
def rk4_odeint(input_, W_a, omega_a, t0, tf, dt):
    def func_(input_, t, W_a, omega_a):
        x, logp_x = input_
        W = W_t(W_a, t)
        omega = omega_t(omega_a, t)

        return (jnp.sum(jnp.fft.ifft2(jnp.fft.fft2(W) * jnp.fft.fft2(jnp.sin(omega.reshape(-1, 1, 1) * x))).real, axis=0),
        jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * (omega.reshape(-1, 1, 1) * jnp.cos(omega.reshape(-1, 1, 1) * x))))

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



