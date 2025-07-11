from util import *
import argparse

parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("--L", type=int, default=6, help="Size of the system")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--train_steps", type=int, default=5000, help="Number of training steps")
parser.add_argument("--num_samples", type=int, default=512, help="Number of samples")
parser.add_argument("--f", type=int, default=20, help="Frequency parameter")
parser.add_argument("--metro_samples", type=int, default=1024, help="Number of metro samples")
parser.add_argument("--t0", type=float, default=0.0, help="Start time")
parser.add_argument("--tf", type=float, default=1.0, help="End time")
parser.add_argument("--dt", type=float, default=0.02, help="Time step")
parser.add_argument("--m2", type=int, default=-4, help="Mass squared term")
parser.add_argument("--lam", type=float, default=6.975, help="Lambda parameter")
parser.add_argument("--t_kernel", type=int, default=21, help="Kernel time")
parser.add_argument("--lr0", type=float, default=2e-4, help="Initial learning rate")
parser.add_argument("--alpha", type=float, default=5e-5, help="Alpha for learning rate decay")
args = parser.parse_args()

L = args.L
seed = args.seed
train_steps = args.train_steps
num_samples = args.num_samples
f = args.f
metro_samples = args.metro_samples
t0, tf, dt = args.t0, args.tf, args.dt
m2 = args.m2
lam =  args.lam
t_kernel = args.t_kernel
lr0 = args.lr0
W_a0 = jnp.zeros((t_kernel, f, L+1, L+1))
alpha = args.alpha
omega_a = jnp.concatenate((0.4*(jnp.arange(f)+0.5)[None, :], jnp.zeros((t_kernel-1, f))), axis = 0)
lr_schedule = optax.cosine_decay_schedule(
    init_value = lr0,
    decay_steps = train_steps,
    alpha = alpha
)

solver = optax.chain( # Adam is a minimization algorithm, so we negate the gradients  # Apply the learning rate schedule
        #optax.clip_by_global_norm(100.0),
        optax.sgd(learning_rate=lr_schedule)  # Use Adam updates to scale the gradients
)
'''
solver = optax.chain(  # Use Adam updates to scale the gradients
    optax.scale_by_schedule(lr_schedule),  # Apply the learning rate schedule
    optax.scale(-1)  # Adam is a minimization algorithm, so we negate the gradients
)
'''
params = (W_a0, omega_a)
opt_state = solver.init(params)
batch_phi4 = jax.jit(vmap(partial(phi4_action, m2=m2, lam=lam), in_axes=(0,)))
batch_diff_phi4 = jax.jit(vmap(partial(diff_phi4_action, m2=m2, lam=lam), in_axes=(0,)))

for i in range(train_steps):
    t0_ = time.time()

    W_a = jnp.transpose(vmap(vmap(initial_mask, (0, None)), (1, None))(W_a0, L), (1, 0, 2, 3))
    x0, logp_x0, dlogp_x0 = get_batch(num_samples, L, seed)
    diff_xf_W_t0 = jnp.zeros_like(jnp.repeat(W_a[None, :], num_samples, axis=0))
    diff_xf_omega_t0 = jnp.zeros_like(jnp.repeat(W_a[None, :], num_samples, axis=0))
    xf, logp_prob, diff_logp_x, int_diff_xf_W, int_diff_xf_omega = rk4_odeint(dt, (
    x0, jnp.zeros(num_samples), dlogp_x0, diff_xf_W_t0, diff_xf_omega_t0), jnp.array([t0, tf]), W_a, omega_a)
    diff_logp_x += batch_diff_phi4(xf)
    grad_w = jnp.mean(jnp.fft.ifft2(
        jnp.flip(jnp.roll(jnp.fft.fft2(int_diff_xf_W), (-1, -1), axis=(-2, -1)), (-2, -1)) * jnp.fft.fft2(diff_logp_x)[
                                                                                             :, None, None]).real,
                      axis=0)
    grad_wa0 = jnp.transpose(vmap(vmap(backward_propagate_grads, (0, None)), (1, None))(grad_w, L), (1, 0, 2, 3))/L**2
    grad_omega = jnp.mean(jnp.sum(diff_logp_x[:, None, None] * int_diff_xf_omega, axis=(-2, -1)), axis=0)/L**2
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
