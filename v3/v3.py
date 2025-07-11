from util import *
from util_path import *
import argparse
@partial(jax.jit, static_argnames=('num_samples','L','t0', 'tf', 'dt'))
def loss_func(params, seed, num_samples, L, t0, tf, dt):
    W_a0, omega_a = params
    W_a = jnp.transpose(vmap(vmap(initial_mask, (0,)), (1,))(W_a0), (1, 0, 2, 3))
    x0, logp_x0, dlogp_x0 = get_batch(num_samples, L, seed)
    xf, logp_prob= rk4_odeint((x0, jnp.zeros(x0.shape[0])), W_a, omega_a, t0, tf, dt)
    logp_xf = logp_x0 - logp_prob
    logp = -batch_phi4(xf)
    logp_x = logp_xf - logp
    jax.lax.cond(seed % 10 == 0, lambda _: jax.debug.print("ess:{}", compute_ess(logp, logp_xf)), lambda _: None, None)

    return jnp.mean(logp_x)

@partial(jax.jit, static_argnames=('num_samples','L','t0', 'tf', 'dt'))
def loss_func_path(params, seed, num_samples, diff_xf_W_t0, diff_xf_omega_t0, L, t0, tf, dt):
    W_a0, omega_f = params
    print("W_a0:", W_a0.shape)
    W_a = jnp.transpose(vmap(vmap(initial_mask, (0,)), (1,))(W_a0), (1, 0, 2, 3))
    print("W_a:", W_a.shape)
    x0, logp_x0, dlogp_x0 = get_batch(num_samples, L, seed)
    xf, logp_prob, diff_logp_x, int_diff_xf_W, int_diff_xf_omega = rk4_odeint_path(
        (x0, jnp.zeros(x0.shape[0]), dlogp_x0, diff_xf_W_t0, diff_xf_omega_t0)
        , W_a, omega_f, t0, tf, dt)

    # Add the gradient of the action on diff_logp_x
    diff_logp_x += batch_diff_phi4(xf)
    # The following two operations are not vmapped, we do it exactly.
    grad_w = jnp.mean(jnp.fft.ifft2(
        jnp.flip(jnp.roll(jnp.fft.fft2(int_diff_xf_W), (-1, -1), axis=(-2, -1)), (-2, -1)) * jnp.fft.fft2(
            diff_logp_x)[:, None, None]).real, axis=0)
    grad_omega = jnp.mean(jnp.sum(diff_logp_x[:, None] * int_diff_xf_omega, axis=(-2, -1)), axis=0)

    grad_wa0 = jnp.transpose(vmap(vmap(backward_propagate_grads, (0,)), (1,))(grad_w), (1, 0, 2, 3))
    logp_xf = logp_x0 - logp_prob
    logp = -batch_phi4(xf)
    logp_x = logp_xf - logp
    ess = compute_ess(logp, logp_xf)
    return logp_x.mean(0), ess, grad_wa0, grad_omega

parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("--L", type=int, default=6, help="Size of the system")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--train_steps", type=int, default=2, help="Number of training steps")
parser.add_argument("--num_samples", type=int, default=8, help="Number of samples")
parser.add_argument("--f", type=int, default=16, help="Frequency parameter")
parser.add_argument("--metro_samples", type=int, default=1024, help="Number of metro samples")
parser.add_argument("--t0", type=float, default=0.0, help="Start time")
parser.add_argument("--tf", type=float, default=1.0, help="End time")
parser.add_argument("--dt", type=float, default=0.02, help="Time step")
parser.add_argument("--m2", type=int, default=-4, help="Mass squared term")
parser.add_argument("--lam", type=float, default=6.975, help="Lambda parameter")
parser.add_argument("--t_kernel", type=int, default=21, help="Kernel time")
parser.add_argument("--lr0", type=float, default=2e-4, help="Initial learning rate")
parser.add_argument("--alpha", type=float, default=2e-5, help="Alpha for learning rate decay")
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
W_a0 = jnp.zeros((t_kernel, f+1, L+1, L+1))
alpha = args.alpha
omega_a = jnp.concatenate((jnp.ones(1), 0.4*(jnp.arange(f)+0.5)), axis = 0)
diff_xf_W_t0 = jnp.zeros((num_samples, t_kernel, f+1, L, L))
diff_xf_omega_t0 = jnp.zeros((num_samples, f+1, L, L))
lr_schedule = optax.cosine_decay_schedule(
    init_value = lr0,
    decay_steps = train_steps,
    alpha = alpha
)

solver = optax.chain( # Adam is a minimization algorithm, so we negate the gradients  # Apply the learning rate schedule
        optax.clip_by_global_norm(50.0),
        optax.sgd(learning_rate=lr_schedule)  # Use Adam updates to scale the gradients
)
params = (W_a0, omega_a)
opt_state = solver.init(params)
batch_phi4 = jax.jit(vmap(partial(phi4_action, m2=m2, lam=lam), in_axes=(0,)))
batch_diff_phi4 = jax.jit(vmap(partial(diff_phi4_action, m2=m2, lam=lam), in_axes=(0,)))
method = "path"

for i in range(train_steps):
    t0_ = time.time()
    if method == "path":
        loss, ess, grad_wa0, grad_omega = loss_func_path(params, seed, num_samples, diff_xf_W_t0, diff_xf_omega_t0, L, t0, tf, dt)
        #print(grad_wa0, grad_omega)
        updates, opt_state = solver.update((grad_wa0, grad_omega), opt_state, params)
        params = optax.apply_updates(params, updates)
        W_a0, omega_f = params

    else:
        loss = loss_func(params, seed, num_samples, L, t0, tf, dt)
        grad = jax.grad(loss_func)(params, seed, num_samples, L, t0, tf, dt)
        updates, opt_state = solver.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

    if i % 1 == 0:
        if method == "path":
            print("ess:", ess)
        print("itert: ", time.time() - t0_)
        print('Iter: {}, loss: {:.4f}\n'.format(i, loss.item()))
        # print("grad_w", grad_w)
        # print("grad_omega", grad_omega)
    seed += 1
