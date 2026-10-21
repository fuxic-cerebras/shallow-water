
"""Script that solves the 2D shallow water equations using finite
differences where the momentum equations are taken to be linear, but the
continuity equation is solved in its nonlinear form. The model supports turning
on/off various terms, but in its most complete form, the model solves the following
set of equations:

    du/dt - fv = -g*d(eta)/dx + tau_x/(rho_0*H)- kappa*u
    dv/dt + fu = -g*d(eta)/dy + tau_y/(rho_0*H)- kappa*v
    d(eta)/dt + d((eta + H)*u)/dx + d((eta + H)*v)/dy = sigma - w

where f = f_0 + beta*y can be the full latitude varying coriolis parameter.
For the momentum equations, an ordinary forward-in-time centered-in-space
scheme is used. However, the coriolis terms is not so trivial, and thus, one
first finds a predictor for u, v and then a corrected value is computed in
order to include the coriolis terms. In the continuity equation, it's used a
forward difference for the time derivative and an upwind scheme for the non-
linear terms. The model is stable under the CFL condition of

    dt <= min(dx, dy)/sqrt(g*H)    and    alpha << 1 (if coriolis is used)

where dx, dy is the grid spacing in the x- and y-direction respectively, g is
the acceleration of gravity and H is the resting depth of the fluid.

The grid resolution is a command line parameter, so the same model can be run at
several resolutions without editing the file:

    python swe.py                    # default 150 x 150 grid
    python swe.py -n 300             # 300 x 300 grid
    python swe.py --nx 200 --ny 100  # different resolution per direction

Note that dt follows from the CFL condition above, so a finer grid means a
smaller time step: the same number of time steps then covers less simulated
time. Use --max-time-step to compensate.

Each run saves snapshots of u, v and eta, the grid and the parameters to
swe_fields_nx<N_x>_ny<N_y>.npz, so runs at different resolutions don't overwrite
each other's output. See save_fields() for the contents."""

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import viz_tools

# ==================================================================================
# ================================ Parameter stuff =================================
# ==================================================================================
# --------------- Physical prameters ---------------
L_x = 1E+6              # Length of domain in x-direction
L_y = 1E+6              # Length of domain in y-direction
g = 9.81                 # Acceleration of gravity [m/s^2]
H = 100                # Depth of fluid [m]
f_0 = 1E-4              # Fixed part ofcoriolis parameter [1/s]
beta = 2E-11            # gradient of coriolis parameter [1/ms]
rho_0 = 1024.0          # Density of fluid [kg/m^3)]
tau_0 = 0.1             # Amplitude of wind stress [kg/ms^2]
use_coriolis = True     # True if you want coriolis force
use_friction = False     # True if you want bottom friction
use_wind = False        # True if you want wind stress
use_beta = True         # True if you want variation in coriolis
use_source = False       # True if you want mass source into the domain
use_sink = False       # True if you want mass sink out of the domain

# --------------- Computational prameters ---------------
# These are the defaults; the grid size and the length of the run can be set on
# the command line (see main() at the bottom of the file).
N_x_default = 150            # Number of grid points in x-direction
N_y_default = 150            # Number of grid points in y-direction
max_time_step_default = 5000  # Total number of time steps in simulation
anim_interval_default = 20    # How often to sample for the animations
sample_interval_default = 1000  # How often to sample for time series


@dataclass
class Grid:
    """The computational grid: number of points, spacing, time step from the CFL
    condition and the coordinate arrays/meshgrids used by the solver and plots."""
    N_x: int
    N_y: int
    dx: float
    dy: float
    dt: float
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray


@dataclass
class Terms:
    """Arrays for the optional terms of the model. Each one is None unless the
    corresponding use_* flag above is enabled. param_string is the human readable
    summary of every parameter that went into the run."""
    kappa: Optional[np.ndarray] = None      # Bottom friction coefficient
    tau_x: Optional[np.ndarray] = None      # Wind stress in x-direction
    tau_y: Optional[np.ndarray] = None      # Wind stress in y-direction
    f: Optional[np.ndarray] = None          # Coriolis parameter
    alpha: Optional[np.ndarray] = None      # Parameter needed for coriolis scheme
    beta_c: Optional[np.ndarray] = None     # Parameter needed for coriolis scheme
    L_R: Optional[float] = None             # Rossby deformation radius
    sigma: Optional[np.ndarray] = None      # Mass source
    w: Optional[np.ndarray] = None          # Mass sink
    param_string: str = ""


@dataclass
class Samples:
    """Everything the time loop hands to the visualization: the final state, the
    sampled fields for the animations and the samples for Hovmuller/spectrum. The
    snap_* lists are the coarser snapshots written to the .npz artifact."""
    u_n: np.ndarray
    v_n: np.ndarray
    eta_n: np.ndarray
    u_list: list = field(default_factory=list)
    v_list: list = field(default_factory=list)
    eta_list: list = field(default_factory=list)
    hm_sample: list = field(default_factory=list)
    ts_sample: list = field(default_factory=list)
    t_sample: list = field(default_factory=list)
    snap_u: list = field(default_factory=list)
    snap_v: list = field(default_factory=list)
    snap_eta: list = field(default_factory=list)
    snap_step: list = field(default_factory=list)


def build_grid(N_x, N_y):
    """Function that builds the grid for a given number of grid points in the x-
    and y-direction. The time step is set from the CFL condition, so it shrinks
    as the grid is refined."""
    dx = L_x/(N_x - 1)                   # Grid spacing in x-direction
    dy = L_y/(N_y - 1)                   # Grid spacing in y-direction
    dt = 0.1*min(dx, dy)/np.sqrt(g*H)    # Time step (defined from the CFL condition)
    x = np.linspace(-L_x/2, L_x/2, N_x)  # Array with x-points
    y = np.linspace(-L_y/2, L_y/2, N_y)  # Array with y-points
    X, Y = np.meshgrid(x, y)             # Meshgrid for plotting
    X = np.transpose(X)                  # To get plots right
    Y = np.transpose(Y)                  # To get plots right
    return Grid(N_x = N_x, N_y = N_y, dx = dx, dy = dy, dt = dt,
        x = x, y = y, X = X, Y = Y)


def build_terms(grid):
    """Function that builds the arrays for every enabled term of the model on the
    given grid, and collects the parameter summary while doing so."""
    N_x, N_y, x, y = grid.N_x, grid.N_y, grid.x, grid.y
    X, Y, dt = grid.X, grid.Y, grid.dt
    terms = Terms()

    param_string = "\n================================================================"
    param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(use_coriolis, use_beta)
    param_string += "\nuse_friction = {}\nuse_wind = {}".format(use_friction, use_wind)
    param_string += "\nuse_source = {}\nuse_sink = {}".format(use_source, use_sink)
    param_string += "\ng = {:g}\nH = {:g}".format(g, H)
    param_string += "\nN_x = {:d}\nN_y = {:d}".format(N_x, N_y)
    param_string += "\ndx = {:.2f} km\ndy = {:.2f} km\ndt = {:.2f} s".format(
        grid.dx, grid.dy, dt)

    # Define friction array if friction is enabled.
    if (use_friction is True):
        kappa_0 = 1/(5*24*3600)
        terms.kappa = np.ones((N_x, N_y))*kappa_0
        #terms.kappa[0, :] = kappa_0
        #terms.kappa[-1, :] = kappa_0
        #terms.kappa[:, 0] = kappa_0
        #terms.kappa[:, -1] = kappa_0
        #terms.kappa[:int(N_x/15), :] = 0
        #terms.kappa[int(14*N_x/15)+1:, :] = 0
        #terms.kappa[:, :int(N_y/15)] = 0
        #terms.kappa[:, int(14*N_y/15)+1:] = 0
        #terms.kappa[int(N_x/15):int(2*N_x/15), int(N_y/15):int(14*N_y/15)+1] = 0
        #terms.kappa[int(N_x/15):int(14*N_x/15)+1, int(N_y/15):int(2*N_y/15)] = 0
        #terms.kappa[int(13*N_x/15)+1:int(14*N_x/15)+1, int(N_y/15):int(14*N_y/15)+1] = 0
        #terms.kappa[int(N_x/15):int(14*N_x/15)+1, int(13*N_y/15)+1:int(14*N_y/15)+1] = 0
        param_string += "\nkappa = {:g}\nkappa/beta = {:g} km".format(
            kappa_0, kappa_0/(beta*1000))

    # Define wind stress arrays if wind is enabled.
    if (use_wind is True):
        terms.tau_x = -tau_0*np.cos(np.pi*y/L_y)
        terms.tau_y = np.zeros(len(y))
        param_string += "\ntau_0 = {:g}\nrho_0 = {:g}".format(tau_0, rho_0)

    # Define coriolis array if coriolis is enabled.
    if (use_coriolis is True):
        if (use_beta is True):
            terms.f = f_0 + beta*y      # Varying coriolis parameter
        else:
            terms.f = f_0*np.ones(len(y))           # Constant coriolis parameter

        terms.L_R = np.sqrt(g*H)/f_0    # Rossby deformation radius (independent of beta)
        terms.alpha = dt*terms.f        # Parameter needed for coriolis scheme
        terms.beta_c = terms.alpha**2/4  # Parameter needed for coriolis scheme

        param_string += "\nf_0 = {:g}".format(f_0)
        param_string += "\nMax alpha = {:g}\n".format(terms.alpha.max())
        param_string += "\nRossby radius: {:.1f} km".format(terms.L_R/1000)
        param_string += "\nRossby radius / domain length: {:g}".format(terms.L_R/L_x)
        # The long Rossby wave only exists on a beta-plane.
        if (use_beta is True):
            c_R = beta*g*H/f_0**2   # Long Rossby wave speed
            param_string += "\nLong Rossby wave speed: {:.3f} m/s".format(c_R)
            param_string += "\nLong Rossby transit time: {:.2f} days".format(
                L_x/(c_R*24*3600))
        param_string += "\n================================================================\n"

    # Define source array if source is enabled.
    if (use_source is True):
        terms.sigma = 0.0001*np.exp(-((X-L_x/2)**2/(2*(1E+5)**2)
            + (Y-L_y/2)**2/(2*(1E+5)**2)))

    # Define sink array if sink is enabled. The sink is uniform and sized to remove
    # exactly as much mass as the source adds, so it needs the source to be defined.
    if (use_sink is True):
        if (use_source is not True):
            raise ValueError("use_sink requires use_source: the sink balances the source")
        terms.w = np.ones((N_x, N_y))*terms.sigma.sum()/(N_x*N_y)

    terms.param_string = param_string
    return terms
# ============================= Parameter stuff done ===============================


def initial_conditions(grid):
    """Function that allocates and returns the initial u, v and eta on the given
    grid. The initial eta is a gaussian bump, see the commented out alternatives
    for other initial states."""
    N_x, N_y, X, Y = grid.N_x, grid.N_y, grid.X, grid.Y

    u_n = np.zeros((N_x, N_y))      # To hold u at current time step
    v_n = np.zeros((N_x, N_y))      # To hold v at current time step

    # Initial conditions for u and v.
    u_n[:, :] = 0.0             # Initial condition for u
    v_n[:, :] = 0.0             # Initial condition for u
    u_n[-1, :] = 0.0            # Ensuring initial u satisfy BC
    v_n[:, -1] = 0.0            # Ensuring initial v satisfy BC

    # Initial condition for eta.
    #eta_n = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
    #eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
    eta_n = np.exp(-((X-L_x/2.7)**2/(2*(0.05E+6)**2) + (Y-L_y/4)**2/(2*(0.05E+6)**2)))
    #eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
    #eta_n[int(6*N_x/8):int(7*N_x/8),int(6*N_y/8):int(7*N_y/8)] = 1.0
    #eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
    #eta_n[:, :] = 0.0

    #viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))
    return u_n, v_n, eta_n


# ==================================================================================
# ========================= Main time loop for simulation ==========================
# ==================================================================================
def run_simulation(grid, terms, max_time_step = max_time_step_default,
        anim_interval = anim_interval_default,
        sample_interval = sample_interval_default,
        save_interval = None):
    """Function that integrates the model forward in time on the given grid, and
    returns the final state together with the sampled fields for plotting. Full
    (u, v, eta) snapshots are kept every save_interval time step for the .npz
    artifact, in addition to the animation frames kept every anim_interval."""
    N_x, N_y, dx, dy, dt = grid.N_x, grid.N_y, grid.dx, grid.dy, grid.dt

    # ==============================================================================
    # =================== Allocating arrays and initial conditions =================
    # ==============================================================================
    u_n, v_n, eta_n = initial_conditions(grid)
    u_np1 = np.zeros((N_x, N_y))    # To hold u at next time step
    v_np1 = np.zeros((N_x, N_y))    # To hold v at next time step
    eta_np1 = np.zeros((N_x, N_y))  # To hold eta at next time step

    # Temporary variables (each time step) for upwind scheme in eta equation
    h_e = np.zeros((N_x, N_y))
    h_w = np.zeros((N_x, N_y))
    h_n = np.zeros((N_x, N_y))
    h_s = np.zeros((N_x, N_y))
    uhwe = np.zeros((N_x, N_y))
    vhns = np.zeros((N_x, N_y))

    if (save_interval is None):
        save_interval = sample_interval

    # Sampling variables.
    samples = Samples(u_n = u_n, v_n = v_n, eta_n = eta_n)
    samples.hm_sample.append(eta_n[:, int(N_y/2)])       # Sample initial eta in middle of domain
    samples.ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])  # Sample initial eta at center of domain
    samples.t_sample.append(0.0)                        # Add initial time to t-samples
    samples.snap_u.append(u_n); samples.snap_v.append(v_n)   # Snapshot of the
    samples.snap_eta.append(eta_n); samples.snap_step.append(0)  # initial state
    # =============== Done with setting up arrays and initial conditions ===========

    time_step = 1               # For counting time loop steps
    t_0 = time.perf_counter()   # For timing the computation loop

    while (time_step < max_time_step):
        # ------------ Computing values for u and v at next time step --------------
        u_np1[:-1, :] = u_n[:-1, :] - g*dt/dx*(eta_n[1:, :] - eta_n[:-1, :])
        v_np1[:, :-1] = v_n[:, :-1] - g*dt/dy*(eta_n[:, 1:] - eta_n[:, :-1])

        # Add friction if enabled.
        if (use_friction is True):
            u_np1[:-1, :] -= dt*terms.kappa[:-1, :]*u_n[:-1, :]
            v_np1[:, :-1] -= dt*terms.kappa[:, :-1]*v_n[:, :-1]

        # Add wind stress if enabled.
        if (use_wind is True):
            u_np1[:-1, :] += dt*terms.tau_x/(rho_0*H)
            v_np1[:, :-1] += dt*terms.tau_y[:-1]/(rho_0*H)

        # Use a corrector method to add coriolis if it's enabled.
        if (use_coriolis is True):
            u_np1[:, :] = (u_np1[:, :] - terms.beta_c*u_n[:, :]
                + terms.alpha*v_n[:, :])/(1 + terms.beta_c)
            v_np1[:, :] = (v_np1[:, :] - terms.beta_c*v_n[:, :]
                - terms.alpha*u_n[:, :])/(1 + terms.beta_c)

        v_np1[:, -1] = 0.0      # Northern boundary condition
        u_np1[-1, :] = 0.0      # Eastern boundary condition
        # -------------------------- Done with u and v -----------------------------

        # --- Computing arrays needed for the upwind scheme in the eta equation.----
        h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
        h_e[-1, :] = eta_n[-1, :] + H

        h_w[0, :] = eta_n[0, :] + H
        h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)

        h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
        h_n[:, -1] = eta_n[:, -1] + H

        h_s[:, 0] = eta_n[:, 0] + H
        h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

        uhwe[0, :] = u_np1[0, :]*h_e[0, :]
        uhwe[1:, :] = u_np1[1:, :]*h_e[1:, :] - u_np1[:-1, :]*h_w[1:, :]

        vhns[:, 0] = v_np1[:, 0]*h_n[:, 0]
        vhns[:, 1:] = v_np1[:, 1:]*h_n[:, 1:] - v_np1[:, :-1]*h_s[:, 1:]
        # ------------------------- Upwind computations done ------------------------

        # ----------------- Computing eta values at next time step ------------------
        eta_np1[:, :] = eta_n[:, :] - dt*(uhwe[:, :]/dx + vhns[:, :]/dy)  # Without source/sink

        # Add source term if enabled.
        if (use_source is True):
            eta_np1[:, :] += dt*terms.sigma

        # Add sink term if enabled.
        if (use_sink is True):
            eta_np1[:, :] -= dt*terms.w
        # ----------------------------- Done with eta ------------------------------

        u_n = np.copy(u_np1)        # Update u for next iteration
        v_n = np.copy(v_np1)        # Update v for next iteration
        eta_n = np.copy(eta_np1)    # Update eta for next iteration

        time_step += 1

        # Samples for Hovmuller diagram and spectrum every sample_interval time step.
        if (time_step % sample_interval == 0):
            samples.hm_sample.append(eta_n[:, int(N_y/2)])          # Sample middle of domain for Hovmuller
            samples.ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])  # Sample center point for spectrum
            samples.t_sample.append(time_step*dt)                   # Keep track of sample times.

        # Full snapshots of the state every save_interval time step for the artifact.
        if (time_step % save_interval == 0):
            samples.snap_u.append(u_n)
            samples.snap_v.append(v_n)
            samples.snap_eta.append(eta_n)
            samples.snap_step.append(time_step)

        # Store eta and (u, v) every anim_interval time step for animations.
        if (time_step % anim_interval == 0):
            print("Time: \t{:.2f} hours".format(time_step*dt/3600))
            print("Step: \t{} / {}".format(time_step, max_time_step))
            print("Mass: \t{}\n".format(np.sum(eta_n)))
            samples.u_list.append(u_n)
            samples.v_list.append(v_n)
            samples.eta_list.append(eta_n)

    print("Main computation loop done!\nExecution time: {:.2f} s".format(
        time.perf_counter() - t_0))

    samples.u_n = u_n
    samples.v_n = v_n
    samples.eta_n = eta_n
    return samples
# ============================= Main time loop done ================================


# ==================================================================================
# ================== Visualizing results by call to external file ==================
# ==================================================================================
def visualize(grid, samples, anim_interval = anim_interval_default,
        filetype = "gif"):
    """Function that turns the sampled fields into the animations shown in the
    README. Uncomment the calls below for the static plots and diagrams."""
    X, Y, dt = grid.X, grid.Y, grid.dt

    #viz_tools.pmesh_plot(X, Y, samples.eta_n, "Final state of surface elevation $\eta$")
    #viz_tools.quiver_plot(X, Y, samples.u_n, samples.v_n, "Final state of velocity field $\mathbf{u}(x,y)$")
    #viz_tools.hovmuller_plot(grid.x, samples.t_sample, samples.hm_sample)
    #viz_tools.plot_time_series_and_ft(samples.t_sample, samples.ts_sample)
    # eta_anim = viz_tools.eta_animation(X, Y, samples.eta_list, anim_interval*dt, "eta", filetype = filetype)
    # The 3D surface version of the same field. These two write surface.gif and
    # velocity.gif, the pair shown in the README. Pass filetype = "mp4" for video,
    # which is much smaller and faster to write but needs ffmpeg.
    eta_surf_anim = viz_tools.eta_animation3D(X, Y, samples.eta_list,
        anim_interval*dt, "surface", filetype = filetype)
    quiv_anim = viz_tools.velocity_animation(X, Y, samples.u_list, samples.v_list,
        anim_interval*dt, "velocity", filetype = filetype)
    return eta_surf_anim, quiv_anim
# ============================ Done with visualization =============================


# ==================================================================================
# ============================== Saving the artifact ===============================
# ==================================================================================
def run_postfix(grid):
    """The nx/ny postfix that keeps the output of runs at different resolutions
    apart, e.g. "nx150_ny150"."""
    return "nx{:d}_ny{:d}".format(grid.N_x, grid.N_y)


def save_fields(grid, terms, samples, filename):
    """Function that writes the (u, v, eta) snapshots to a compressed .npz file,
    together with everything needed to interpret them later: the times of the
    snapshots, the coordinate arrays, the grid and physical parameters, the enabled
    term arrays and the Hovmuller/time series samples. Load it with

        data = np.load("<filename>")
        eta = data["eta"]     # shape (n_snapshots, N_x, N_y)
        t = data["t"]         # time [s] of each snapshot
    """
    step = np.array(samples.snap_step, dtype = int)
    fields = {
        # ---- The state at each saved time step, shape (n_snapshots, N_x, N_y) ----
        "u": np.array(samples.snap_u),
        "v": np.array(samples.snap_v),
        "eta": np.array(samples.snap_eta),
        "step": step,                       # Time step index of each snapshot
        "t": step*grid.dt,                  # Time [s] of each snapshot
        # ---- Final state, i.e. after the last time step of the run ----
        "u_final": samples.u_n,
        "v_final": samples.v_n,
        "eta_final": samples.eta_n,
        # ---- The grid ----
        "N_x": grid.N_x, "N_y": grid.N_y,
        "dx": grid.dx, "dy": grid.dy, "dt": grid.dt,
        "x": grid.x, "y": grid.y, "X": grid.X, "Y": grid.Y,
        # ---- Physical parameters and the terms that were enabled ----
        "L_x": L_x, "L_y": L_y, "g": g, "H": H,
        "f_0": f_0, "beta": beta, "rho_0": rho_0, "tau_0": tau_0,
        "use_coriolis": use_coriolis, "use_beta": use_beta,
        "use_friction": use_friction, "use_wind": use_wind,
        "use_source": use_source, "use_sink": use_sink,
        # ---- Hovmuller and time series samples ----
        "hm_sample": np.array(samples.hm_sample),
        "ts_sample": np.array(samples.ts_sample),
        "t_sample": np.array(samples.t_sample),
        "param_string": terms.param_string,
    }

    # Include the arrays of the optional terms that are actually in use.
    for name in ("kappa", "tau_x", "tau_y", "f", "L_R", "sigma", "w"):
        value = getattr(terms, name)
        if (value is not None):
            fields[name] = value

    np.savez_compressed(filename, **fields)
    print("Saved {} snapshots of (u, v, eta) on the {} x {} grid to {}".format(
        len(step), grid.N_x, grid.N_y, filename))
# =========================== Done saving the artifact =============================


def grid_points(value):
    """argparse type for a number of grid points: at least 2 points are needed in
    each direction for the grid spacing to be defined."""
    N = int(value)
    if (N < 2):
        raise argparse.ArgumentTypeError(
            "number of grid points must be at least 2, got {}".format(N))
    return N


def parse_args(argv = None):
    parser = argparse.ArgumentParser(
        description = "Solve the 2D shallow water equations on an N_x by N_y grid.",
        epilog = "Note that dt follows from the CFL condition, so a finer grid means "
            "a smaller time step: the same number of time steps then covers less "
            "simulated time. Use --max-time-step to compensate.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--grid-points", type = grid_points, metavar = "N",
        help = "number of grid points in both directions, shorthand for --nx N --ny N")
    parser.add_argument("--nx", type = grid_points, default = N_x_default,
        help = "number of grid points in the x-direction")
    parser.add_argument("--ny", type = grid_points, default = N_y_default,
        help = "number of grid points in the y-direction")
    parser.add_argument("--max-time-step", type = int, default = max_time_step_default,
        help = "total number of time steps in the simulation")
    parser.add_argument("--anim-interval", type = int, default = anim_interval_default,
        help = "number of time steps between each frame of the animations")
    parser.add_argument("--sample-interval", type = int, default = sample_interval_default,
        help = "number of time steps between each Hovmuller/time series sample")
    parser.add_argument("--save-interval", type = int, default = None,
        help = "number of time steps between each (u, v, eta) snapshot in the .npz "
            "artifact, defaults to --sample-interval; pass the same value as "
            "--anim-interval to save every animation frame")
    parser.add_argument("--save-prefix", default = "swe_fields", metavar = "PREFIX",
        help = "prefix of the .npz artifact, which is written as "
            "PREFIX_nx<N_x>_ny<N_y>.npz")
    parser.add_argument("--no-save", action = "store_true",
        help = "do not write the .npz artifact")
    parser.add_argument("--filetype", choices = ["gif", "mp4"], default = "gif",
        help = "animation format, mp4 is smaller and faster but needs ffmpeg")
    parser.add_argument("--no-viz", action = "store_true",
        help = "run the simulation only, without writing any animation")
    args = parser.parse_args(argv)

    # -n/--grid-points is just a shorthand for setting both directions at once.
    if (args.grid_points is not None):
        args.nx = args.grid_points
        args.ny = args.grid_points
    return args


def main(argv = None):
    args = parse_args(argv)

    grid = build_grid(args.nx, args.ny)
    terms = build_terms(grid)
    postfix = run_postfix(grid)     # Keeps runs at different resolutions apart

    # Write all parameters out to file.
    with open("param_output_{}.txt".format(postfix), "w") as output_file:
        output_file.write(terms.param_string)

    print(terms.param_string)   # Also print parameters to screen

    samples = run_simulation(grid, terms, max_time_step = args.max_time_step,
        anim_interval = args.anim_interval, sample_interval = args.sample_interval,
        save_interval = args.save_interval)

    # Save the fields as an .npz artifact, named after the grid it was run on.
    if (args.no_save is not True):
        save_fields(grid, terms, samples,
            "{}_{}.npz".format(args.save_prefix, postfix))

    if (args.no_viz is True):
        return

    print("\nVisualizing results...")
    anims = visualize(grid, samples, anim_interval = args.anim_interval,
        filetype = args.filetype)
    print("\nVisualization done!")
    plt.show()


if (__name__ == "__main__"):
    main()
