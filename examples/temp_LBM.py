from functools import partial

import jax
import jax.numpy as jnp

# -------------------------
# LBM setup (D2Q9 lattice)
# -------------------------

CXS = jnp.array([0, 1, 0, -1, 0,  1, -1, -1,  1])
CYS = jnp.array([0, 0, 1,  0, -1,  1,  1, -1, -1])

# Weight coefficient
WS = jnp.array([4/9,
                1/9, 1/9, 1/9, 1/9,
                1/36, 1/36, 1/36, 1/36])

# What is this?
OPP = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# -----------------------------------
# Domain and physical/LBM parameters
# -----------------------------------

NX = 180   # streamwise
NY = 80    # cross-stream

OBST_SIZE = 20

obst_x_start = NX // 3
obst_y_start = NY // 2 - OBST_SIZE // 2

U_IN = 0.1
Re = 100.0
L_char = float(OBST_SIZE)

nu = U_IN * L_char / Re
tau = 3.0 * nu + 0.5
omega_relax = 1.0 / tau   # rename to avoid clash with vorticity

print(f"nu={nu:.5f}, tau={tau:.5f} (>0.5), omega_relax={omega_relax:.5f}")

# -------------------------
# Helper functions
# -------------------------

def equilibrium(rho, ux, uy):
    rho_ = rho[..., None]
    ux_ = ux[..., None]
    uy_ = uy[..., None]

    cu = 3.0 * (CXS * ux_ + CYS * uy_)
    u_sq = ux**2 + uy**2
    u_sq_ = u_sq[..., None]

    feq = WS * rho_ * (1.0 + cu + 0.5 * cu**2 - 1.5 * u_sq_)
    return feq


def macroscopic(f):
    rho = jnp.sum(f, axis=2)
    ux = jnp.sum(f * CXS, axis=2) / rho
    uy = jnp.sum(f * CYS, axis=2) / rho
    return rho, ux, uy


def stream(f):
    def stream_dir(fi, cx, cy):
        return jnp.roll(jnp.roll(fi, cy, axis=0), cx, axis=1)
    return jax.vmap(stream_dir, in_axes=(2, 0, 0), out_axes=2)(f, CXS, CYS)


def build_solid_mask():
    solid = jnp.zeros((NY, NX), dtype=bool)
    # top & bottom walls
    solid = solid.at[0, :].set(True)
    solid = solid.at[NY - 1, :].set(True)
    # square obstacle
    solid = solid.at[
        obst_y_start:obst_y_start + OBST_SIZE,
        obst_x_start:obst_x_start + OBST_SIZE
    ].set(True)
    return solid


SOLID_MASK = build_solid_mask()


def bounce_back(f, solid_mask):
    f_old = f
    f_new = jnp.zeros_like(f_old)
    for i in range(9):
        f_new = f_new.at[..., i].set(
            jnp.where(solid_mask, f_old[..., OPP[i]], f_old[..., i])
        )
    return f_new


def inlet_bc_zou_he(f, ux_in):
    x = 0
    f0 = f[:, x, 0]
    f2 = f[:, x, 2]
    f4 = f[:, x, 4]
    f3 = f[:, x, 3]
    f6 = f[:, x, 6]
    f7 = f[:, x, 7]

    ux = ux_in
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)

    f1 = f3 + (2.0 / 3.0) * rho * ux
    f5 = f7 + 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * ux
    f8 = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux

    f = f.at[:, x, 1].set(f1)
    f = f.at[:, x, 5].set(f5)
    f = f.at[:, x, 8].set(f8)
    return f


def outlet_bc_zero_gradient(f):
    return f.at[:, -1, :].set(f[:, -2, :])


# -------------------------
# LBM step
# -------------------------

@jax.jit
def lbm_step(f):
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_post = f - omega_relax * (f - feq)
    f_streamed = stream(f_post)
    f_bb = bounce_back(f_streamed, SOLID_MASK)
    f_in = inlet_bc_zou_he(f_bb, U_IN)
    f_out = outlet_bc_zero_gradient(f_in)
    return f_out


# ===================================
# NEW: Vorticity and streamfunction
# ===================================

def compute_vorticity(ux, uy, dx=1.0, dy=1.0):
    """
    ω = dv/dx - du/dy using 2nd-order central differences with periodic x.
    (NY, NX) → (NY, NX)
    Near top/bottom walls you’ll get artifacts; usually you ignore
    the first/last row in plots.
    """
    dv_dx = (jnp.roll(uy, -1, axis=1) - jnp.roll(uy, 1, axis=1)) / (2.0 * dx)
    du_dy = (jnp.roll(ux, -1, axis=0) - jnp.roll(ux, 1, axis=0)) / (2.0 * dy)
    return dv_dx - du_dy


def solve_streamfunction_poisson(omega, solid_mask, n_iter=400, dx=1.0, dy=1.0):
    """
    Solve Δψ = -ω on fluid cells with ψ=0 on solid/walls via Jacobi.
    Periodic in x, Dirichlet ψ=0 on SOLID_MASK.
    """
    psi = jnp.zeros_like(omega)
    dx2 = dx * dx
    dy2 = dy * dy
    coef = 1.0 / (2.0 * (dx2 + dy2))

    def body_fun(k, psi_cur):
        psi_e = jnp.roll(psi_cur, -1, axis=1)
        psi_w = jnp.roll(psi_cur,  1, axis=1)
        psi_n = jnp.roll(psi_cur, -1, axis=0)
        psi_s = jnp.roll(psi_cur,  1, axis=0)

        rhs = -omega
        psi_new = coef * (
            (psi_e + psi_w) * dy2 +
            (psi_n + psi_s) * dx2 -
            rhs * dx2 * dy2
        )

        # Enforce ψ=0 on solids/walls
        psi_new = jnp.where(solid_mask, 0.0, psi_new)
        return psi_new

    psi = jax.lax.fori_loop(0, n_iter, body_fun, psi)
    return psi


# ===================================
# NEW: JAX trajectory-style wrapper
# ===================================

def make_initial_state():
    rho0 = jnp.ones((NY, NX))
    ux0 = U_IN * jnp.ones((NY, NX))
    uy0 = jnp.zeros((NY, NX))
    f0 = equilibrium(rho0, ux0, uy0)
    f0 = bounce_back(f0, SOLID_MASK)  # obstacle initially filled as solid
    return f0


@partial(jax.jit, static_argnames=["n_steps", "record_interval"])
def lbm_trajectory(n_steps, record_interval, stream_iters=300):
    """
    JAX-style trajectory:
      - Evolves f for n_steps
      - Records (rho, ux, uy, omega, psi) every `record_interval` steps
      - Returns: final f, and history as tuple of arrays:
          (rho_hist, ux_hist, uy_hist, omega_hist, psi_hist)
        each with shape (n_records, NY, NX)
    """
    f0 = make_initial_state()
    n_records = n_steps // record_interval

    def one_chunk(f, _):
        # advance by record_interval steps
        def inner_step(f_inner, _):
            return lbm_step(f_inner), None

        f_new, _ = jax.lax.scan(inner_step, f, None,
                                length=record_interval)

        # macroscopic + post-processing
        rho, ux, uy = macroscopic(f_new)
        omega_field = compute_vorticity(ux, uy)
        psi_field = solve_streamfunction_poisson(
            omega_field, SOLID_MASK, n_iter=stream_iters
        )

        snapshot = (rho, ux, uy, omega_field, psi_field)
        return f_new, snapshot

    f_final, snapshots = jax.lax.scan(
        one_chunk, f0, None, length=n_records
    )
    # snapshots is a tuple of 5 arrays, each (n_records, NY, NX)
    return f_final, snapshots


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # Example: run 20k steps, record every 1000
    n_steps = 20000
    record_interval = 1000

    f_final, (rho_hist, ux_hist, uy_hist, omega_hist, psi_hist) = \
        lbm_trajectory(n_steps, record_interval, stream_iters=300)

    # Take last snapshot for plotting
    rho = rho_hist[-1]
    ux = ux_hist[-1]
    uy = uy_hist[-1]
    omega = omega_hist[-1]
    psi = psi_hist[-1]

    # Example plotting (outside JAX, with matplotlib):
    import matplotlib.pyplot as plt
    
    speed = jnp.sqrt(ux**2 + uy**2)
    
    plt.figure()
    plt.imshow(ux, origin="lower")
    plt.colorbar(label="|u|")
    plt.contour(SOLID_MASK, levels=[0.5], colors="k")
    plt.title("Ux")
    plt.savefig("ux.png")

    plt.figure()
    plt.imshow(uy, origin="lower")
    plt.colorbar(label="|u|")
    plt.contour(SOLID_MASK, levels=[0.5], colors="k")
    plt.title("Uy")
    plt.savefig("uy.png")

    plt.figure()
    plt.imshow(speed, origin="lower")
    plt.colorbar(label="|u|")
    plt.contour(SOLID_MASK, levels=[0.5], colors="k")
    plt.title("Speed")
    plt.savefig("velocity.png")
    
    plt.figure()
    plt.imshow(omega, origin="lower")
    plt.colorbar(label="vorticity")
    plt.contour(SOLID_MASK, levels=[0.5], colors="k")
    plt.title("Vorticity")
    plt.savefig("vorticity.png")
    
    plt.figure()
    plt.imshow(psi, origin="lower")
    plt.colorbar(label="streamfunction ψ")
    plt.contour(SOLID_MASK, levels=[0.5], colors="k")
    plt.title("Streamfunction")
    plt.savefig("streamfunction.png")
