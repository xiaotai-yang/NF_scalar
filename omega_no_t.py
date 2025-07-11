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


def initial_mask(A, L):
    A += jnp.transpose(A, (1, 0))
    A += A[::-1, :]
    A += A[:, ::-1]
    A/= 8
    return A[:-1, :-1]


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
    key = PRNGKey(seed)
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
        omega = omega_a
        #omega = omega_t(omega_a, t)
        t_k = W_a.shape[0]//2
        a_w = jnp.concatenate((jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        #a_omega = jnp.concatenate((jnp.array([1., ]), jnp.sin((jnp.arange(t_k) + 1) * t), jnp.cos((jnp.arange(t_k) + 1) * t)), axis=0)
        return (
        jnp.sum(jnp.fft.ifft2(jnp.fft.fft2(W) * jnp.fft.fft2(jnp.sin(omega.reshape(-1, 1, 1) * x))).real, axis = 0),
        jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * (omega.reshape(-1, 1, 1) * jnp.cos(omega.reshape(-1, 1, 1) * x))),
        -jnp.sum(omega.reshape(-1, 1, 1) * jnp.cos(omega.reshape(-1, 1, 1) * x) * jnp.fft.ifft2(
        jnp.flip(jnp.roll(jnp.fft.fft2(W), (-1, -1), (-1, -2)), (-1, -2)) * jnp.fft.fft2(d_logp_x)).real,
        axis=0) + jnp.sum(W[:, 0, 0].reshape(-1, 1, 1) * ((omega ** 2).reshape(-1, 1, 1) * jnp.sin(omega.reshape(-1, 1, 1) * x)), axis=0),
        a_w.reshape(-1, 1, 1, 1) * jnp.sin(omega.reshape(-1, 1, 1) * x),
        jnp.fft.ifft2(jnp.fft.fft2(W)*jnp.fft.fft2((x*jnp.cos(omega.reshape(-1, 1, 1)*x)))).real)

    """Integrate a system of ODEs using the 4th order Runge-Kutta method."""

    def step_func(cur_y, cur_t, dt):
        """Take one step of RK4."""
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


L = 4
lr0 = 0.002
lr_schedule = optax.cosine_decay_schedule(
    init_value = lr0,
    decay_steps = 10000,
    alpha = 1e-5
)
seed = 0
train_steps = 500
num_samples = 256
f = 9
metro_samples = 1000
t0, tf, dt = 0., 1., 0.02
m2 = -1
lam = 1
t_kernel = 15
W_a0 = jnp.zeros((t_kernel, f, L+1, L+1))
omega_a = jnp.arange(f)+0.5
solver = optimizer = optax.chain(
    optax.scale_by_adam(b1=0.8, b2=0.9),  # Use Adam updates to scale the gradients
    optax.scale_by_schedule(lr_schedule),  # Apply the learning rate schedule
    optax.scale(-1)  # Adam is a minimization algorithm, so we negate the gradients
)
params = (W_a0, omega_a)
opt_state = solver.init(params)
batch_phi4 = jax.jit(vmap(partial(phi4_action, m2=m2, lam=lam), in_axes=(0,)))
batch_diff_phi4 = jax.jit(vmap(partial(diff_phi4_action, m2=m2, lam=lam), in_axes=(0,)))

for i in range(train_steps):
    t0_ = time.time()

    W_a = jnp.transpose(vmap(vmap(initial_mask, (0, None)), (1, None))(W_a0, L), (1, 0, 2, 3))
    x0, logp_x0, dlogp_x0 = get_batch(num_samples, L, seed)
    diff_xf_W_t0 = jnp.zeros_like(jnp.repeat(W_a[None, :], num_samples, axis=0))
    diff_xf_omega_t0 = jnp.zeros((num_samples, f, L, L))
    xf, logp_prob, diff_logp_x, int_diff_xf_W, int_diff_xf_omega = rk4_odeint(dt, (
    x0, jnp.zeros(num_samples), dlogp_x0, diff_xf_W_t0, diff_xf_omega_t0), jnp.array([t0, tf]), W_a, omega_a)
    diff_logp_x += batch_diff_phi4(xf)
    grad_w = jnp.mean(jnp.fft.ifft2(
        jnp.flip(jnp.roll(jnp.fft.fft2(int_diff_xf_W), (-1, -1), axis=(-2, -1)), (-2, -1)) * jnp.fft.fft2(diff_logp_x)[
        :, None, None]).real, axis=0)
    grad_wa0 = jnp.transpose(vmap(vmap(backward_propagate_grads, (0, None)), (1, None))(grad_w, L), (1, 0, 2, 3))
    grad_omega = jnp.mean(jnp.sum(diff_logp_x[:, None] * int_diff_xf_omega, axis=(-2, -1)), axis=0)
    logp_xf = logp_x0 - logp_prob
    logp = -batch_phi4(xf)
    logp_x = logp_xf - logp
    loss = logp_x.mean(0)

    # print("W", W.shape, "omega", omega.shape)
    # print("grad_w", grad_w.shape, "grad_omega", grad_omega.shape)
    updates, opt_state = solver.update((grad_wa0, grad_omega), opt_state, params)
    params = optax.apply_updates(params, updates)
    W_a0, omega_a = params
    if i % 10 == 0:
        print("ess:", compute_ess(logp, logp_xf))
        print("itert: ", time.time() - t0_)
        print('Iter: {}, loss: {:.4f}\n'.format(i, loss.item()))
        # print("grad_w", grad_w)
        # print("grad_omega", grad_omega)
    seed += 1
    # print(jnp.linalg.norm(W-W_a), jnp.linalg.norm(omega-omega_a))