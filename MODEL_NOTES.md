# Shallow-water model — notes on the equations and the code

Reference notes for `swe.py` / `viz_tools.py`: what is being solved, how each
term is discretized, and why the schemes are arranged the way they are.

1. [Governing equations](#1-governing-equations) — the equations as written
2. [What the equations mean](#2-what-the-equations-mean) — physical reading, term by term
3. [Grid and staggering](#3-grid-and-staggering)
4. [Discretization](#4-discretization) — derivation of the update rules, the schemes, stability
5. [Derived scales](#5-derived-scales-reported-at-startup)
6. [Code layout](#6-code-layout)
7. [Configuration switches](#7-configuration-switches-swepy42-47)
8. [Fixed issues](#8-fixed-issues)
9. [Running it](#9-running-it)

Sections 1–2 are physics; 3–4 are numerics. If you only want to understand what
the model *does*, section 2 is the one to read.

---

## 1. Governing equations

The model integrates the 2D shallow-water equations on a rotating plane in a
closed rectangular basin. The prognostic variables are the surface elevation
$\eta(x,y,t)$ and the depth-averaged horizontal velocities $u(x,y,t)$,
$v(x,y,t)$, above a resting depth $H$.

The **momentum equations are linearized** (no advection of momentum):

$$
\frac{\partial u}{\partial t} - f v
= -g \frac{\partial \eta}{\partial x}
+ \frac{\tau_x}{\rho_0 H}
- \kappa u
$$

$$
\frac{\partial v}{\partial t} + f u
= -g \frac{\partial \eta}{\partial y}
+ \frac{\tau_y}{\rho_0 H}
- \kappa v
$$

The **continuity equation is kept fully nonlinear** — the transport uses the
total layer thickness $\eta + H$, not just $H$:

$$
\frac{\partial \eta}{\partial t}
+ \frac{\partial}{\partial x}\big[(\eta + H)\,u\big]
+ \frac{\partial}{\partial y}\big[(\eta + H)\,v\big]
= \sigma - w
$$

with

| symbol | meaning | code |
|---|---|---|
| $f = f_0 + \beta y$ | $\beta$-plane Coriolis parameter | `f_0`, `beta`, `use_beta` |
| $g$ | gravitational acceleration (or reduced gravity) | `g` |
| $H$ | resting depth | `H` |
| $\kappa$ | linear bottom-friction coefficient | `kappa_0`, `use_friction` |
| $\tau_x,\tau_y$ | surface wind stress | `tau_0`, `use_wind` |
| $\rho_0$ | fluid density | `rho_0` |
| $\sigma$ | mass source | `use_source` |
| $w$ | mass sink | `use_sink` |

This split is deliberate: dropping $\mathbf{u}\cdot\nabla\mathbf{u}$ keeps the
wave dynamics clean and linear, while the nonlinear continuity equation makes
mass transport exact, so total mass is conserved to roundoff in a closed basin.

### Boundary conditions

No flow through the walls of the basin:

$$
u = 0 \quad \text{at } x = \pm L_x/2,
\qquad
v = 0 \quad \text{at } y = \pm L_y/2
$$

### Initial condition

As shipped, an off-center Gaussian bump in surface elevation with
$\sigma_b = 50\ \mathrm{km}$ (`swe.py:156`):

$$
\eta(x,y,0) = \exp\!\left[
-\frac{\left(x - \tfrac{L_x}{2.7}\right)^2}{2\sigma_b^2}
-\frac{\left(y - \tfrac{L_y}{4}\right)^2}{2\sigma_b^2}
\right],
\qquad u = v = 0
$$

Waves radiate outward at the gravity-wave speed $\sqrt{gH}$, reflect off the
walls, and are progressively turned by rotation.

---

## 2. What the equations mean

A physical reading of §1, term by term — no numerics.

### 2.1 The three fields

| field | is | units |
|---|---|---|
| $u$ | eastward velocity ($x$-direction) | m/s |
| $v$ | northward velocity ($y$-direction) | m/s |
| $\eta$ | surface elevation above resting level | m |

$u$ and $v$ are **depth-averaged**: that is the shallow-water approximation
itself. The layer is thin compared with the horizontal scales of motion, so the
horizontal flow is taken as uniform from seabed to surface — no $z$ dependence,
and no prognostic vertical velocity anywhere in the model. The surface simply
rises and falls as $\eta$.

$\eta$ is measured from the resting level, so $\eta > 0$ is water piled up,
$\eta < 0$ a depression, and the **actual local depth** is $H + \eta$. That sum
is what the transport terms use. In the default run $\eta$ starts as a 1 m bump
on a 100 m layer — a 1% perturbation, which is what justifies linearizing the
momentum equations while keeping continuity nonlinear.

### 2.2 Momentum: $F = ma$ for a water column

$$
\frac{\partial u}{\partial t} - f v
= -g \frac{\partial \eta}{\partial x}
+ \frac{\tau_x}{\rho_0 H}
- \kappa u
$$

Newton's second law per unit mass, for a vertical column of water. Left side is
acceleration, right side the forces.

**$-g\,\partial\eta/\partial x$ — pressure gradient.** The engine of the whole
system, and the only active force in the default configuration. In a shallow
layer, pressure at any depth is just the weight of the water above it, so
pressure differences reduce *entirely* to differences in surface height. Water
accelerates downhill, away from where the surface is high.

**$-fv$, and $+fu$ in the $y$ equation — Coriolis.** Not a real force; it is the
bookkeeping cost of working in a rotating frame. Its structure matters more than
its magnitude: the pair $(-fv, +fu)$ is a **rotation operator**. It changes the
direction of the flow while doing no work on it, deflecting moving water to the
right when $f > 0$ (northern hemisphere). Acting alone it sends a parcel in
circles — inertial oscillations at frequency $f$. This is exactly why the
numerical treatment of these terms must be a pure rotation with no amplitude
change (§4.3).

**$\tau_x/(\rho_0 H)$ — wind stress.** A force applied at the surface, but the
column moves as a unit, so it is spread over the full depth — hence dividing by
$H$. Deeper water responds more sluggishly to the same wind.

**$-\kappa u$ — bottom friction.** Linear drag: a decay term pulling the flow
back toward rest with an e-folding time $1/\kappa$ (five days as configured).

**What is absent** is the advection term $\mathbf{u}\cdot\nabla\mathbf{u}$ — a
parcel carrying its own momentum along. Its absence is what "linearized
momentum" means. Dropping it is defensible when velocities are weak enough that
the flow pattern evolves by wave propagation rather than by being swept along by
itself. It costs eddies, jets and turbulence, and buys clean wave dynamics.

### 2.3 Continuity: bookkeeping for water

$$
\frac{\partial \eta}{\partial t}
+ \frac{\partial}{\partial x}\big[(\eta + H)u\big]
+ \frac{\partial}{\partial y}\big[(\eta + H)v\big]
= \sigma - w
$$

Not a force balance at all — conservation of volume, and the origin of the
surface height. Water is incompressible, so the only way for the surface to rise
somewhere is for more water to flow in than out.

Writing $h = \eta + H$ for the actual depth makes it transparent:

$$
\frac{\partial h}{\partial t} = -\nabla\cdot(h\mathbf{u})
$$

$h\mathbf{u}$ is the **volume transport** — how much water crosses a unit width
per second, velocity times the depth it flows through. Its divergence is the net
outflow from a column: net outflow drains the column and the surface drops, net
inflow raises it. $\sigma$ and $w$ are external plumbing, adding or removing
water irrespective of the flow.

The nonlinearity kept here is precisely $\eta u$, a product of two unknowns. The
linearized alternative would use $H\mathbf{u}$, i.e. pretend transport happens
through the resting depth. Keeping $(\eta+H)\mathbf{u}$ means a crest — where
the water is deeper — transports more efficiently than a trough, so crests
travel slightly faster and steepen. It also makes mass conservation exact rather
than approximate, which matters over a long integration.

### 2.4 Why the two equations need each other

Neither does anything alone. Together they are a feedback loop:

> a surface tilt accelerates the flow (momentum) → the flow converges somewhere
> (continuity) → convergence raises the surface there → which creates a new tilt
> → which accelerates the flow the other way.

Each stage overshoots, and that oscillation *is* a wave. It falls out in two
lines. Take one dimension, no rotation, no friction, small amplitude:

$$
\frac{\partial u}{\partial t} = -g\frac{\partial \eta}{\partial x},
\qquad
\frac{\partial \eta}{\partial t} = -H\frac{\partial u}{\partial x}
$$

Differentiate the first in $x$, the second in $t$, and eliminate $u$:

$$
\frac{\partial^2 \eta}{\partial t^2} = gH\,\frac{\partial^2 \eta}{\partial x^2}
$$

The classic wave equation, with speed $c = \sqrt{gH} \approx 31\ \mathrm{m/s}$
for a 100 m layer. That number is why $\Delta t$ has to be tens of seconds:
information crosses a grid cell that fast, and the numerics cannot fall behind
it (§4.5).

### 2.5 What rotation adds

Restoring $f$ and repeating the elimination gives the dispersion relation for
inertia–gravity (Poincaré) waves:

$$
\omega^2 = f^2 + gH k^2
$$

Two regimes fall out, both visible in the animations:

- **Short waves** — $k$ large, scales below the Rossby radius
  $L_R = \sqrt{gH}/f$. The $gHk^2$ term dominates, rotation barely matters, and
  these are ordinary gravity waves radiating outward: the fast ripples fleeing
  the initial bump.
- **Long waves** — $k$ small, scales above $L_R$. Here $\omega \to f$, a *floor*
  on the frequency: nothing oscillates more slowly than the rotation rate. What
  remains instead is near-**geostrophic balance**, in which acceleration nearly
  vanishes and the two surviving terms stand off against one another:

$$
f v \approx g \frac{\partial \eta}{\partial x}
$$

In that balance the flow runs *along* surface contours instead of downhill —
around the bump rather than away from it. This is the balance most large-scale
ocean and atmospheric flow sits in, and it is why the residual blob in the
animation persists and rotates rather than simply dispersing.

Finally, letting $f$ vary with latitude ($f = f_0 + \beta y$) breaks that
balance very slowly: a column displaced north or south finds itself with the
wrong rotation rate and must adjust. That slow westward adjustment is a
**Rossby wave**, at $c_R = \beta g H / f_0^2 \approx 2\ \mathrm{m/s}$ — some
fifteen times slower than the gravity waves, which is why seeing them needs
either a long run or the reduced-gravity parameter set in `scenarios.txt`.

---

## 3. Grid and staggering

`swe.py:64-66` builds the meshgrid and then transposes it, so **all arrays are
indexed `[x, y]`**: `A[i, j]` corresponds to $(x_i, y_j)$. This is why
`eta_n[:, N_y//2]` is a west–east section through the middle of the domain.

The variables live on an **Arakawa C-grid** — staggered, though the staggering
is expressed only through array slicing:

$$
\eta_{i,j} \ \text{at cell centers},
\qquad
u_{i+\frac12,\,j} \ \text{at east faces},
\qquad
v_{i,\,j+\frac12} \ \text{at north faces}
$$

The code-to-math mapping is

$$
\texttt{u\_n[i,j]} \equiv u^n_{i+\frac12,\,j},
\qquad
\texttt{v\_n[i,j]} \equiv v^n_{i,\,j+\frac12},
\qquad
\texttt{eta\_n[i,j]} \equiv \eta^n_{i,j}
$$

You can read the staggering directly off the pressure-gradient term: a
difference of two cell-centered $\eta$ values naturally lands on the face
between them.

Grid spacing and time step (`swe.py:57-59`):

$$
\Delta x = \frac{L_x}{N_x - 1},
\qquad
\Delta y = \frac{L_y}{N_y - 1},
\qquad
\Delta t = 0.1\,\frac{\min(\Delta x, \Delta y)}{\sqrt{gH}}
$$

---

## 4. Discretization

### 4.1 Derivation: from the PDEs to the update rules

Every line of the time loop follows from three mechanical steps — replace the
time derivative with a difference over $\Delta t$, replace the space derivatives
with differences on the staggered grid, then solve for the new time level.
Carrying that out explicitly for the momentum equation (`swe.py:181`):

**Step 1 — discretize time.** A forward (explicit Euler) difference,

$$
\left.\frac{\partial u}{\partial t}\right|^{\,n}
\approx \frac{u^{n+1} - u^{n}}{\Delta t} + O(\Delta t)
$$

turns the momentum equation into

$$
\frac{u^{n+1}_{i+\frac12,\,j} - u^{n}_{i+\frac12,\,j}}{\Delta t}
= -g \left.\frac{\partial \eta}{\partial x}\right|^{\,n}_{i+\frac12,\,j}
$$

Every term on the right is evaluated at the *known* level $n$, which is what
makes the step explicit — no system to solve.

**Step 2 — discretize space.** The gradient is needed at the $u$ point,
$x_{i+\frac12}$. Taylor-expand the two neighbouring cell-centred values about
that point:

$$
\eta_{i+1} = \eta_{i+\frac12}
+ \frac{\Delta x}{2}\eta_x
+ \frac{\Delta x^2}{8}\eta_{xx}
+ \frac{\Delta x^3}{48}\eta_{xxx} + \dots
$$

$$
\eta_{i} = \eta_{i+\frac12}
- \frac{\Delta x}{2}\eta_x
+ \frac{\Delta x^2}{8}\eta_{xx}
- \frac{\Delta x^3}{48}\eta_{xxx} + \dots
$$

Subtracting cancels every even-order term, because the evaluation point is
exactly midway between the two:

$$
\frac{\eta_{i+1} - \eta_{i}}{\Delta x}
= \eta_x + \frac{\Delta x^2}{24}\eta_{xxx} + \dots
= \left.\frac{\partial \eta}{\partial x}\right|_{i+\frac12} + O(\Delta x^2)
$$

So what *looks* like a one-sided difference is in fact centered, and
second-order accurate. This is the payoff of the C-grid staggering: centered
accuracy from the two-point stencil, with no averaging and no extra storage.

**Step 3 — solve for the new level.** Multiply through by $\Delta t$:

$$
\boxed{\;
u^{*}_{i+\frac12,\,j} = u^{n}_{i+\frac12,\,j}
- \frac{g\,\Delta t}{\Delta x}\left(\eta^n_{i+1,j} - \eta^n_{i,j}\right)
\;}
$$

which is `swe.py:181` verbatim, for $i = 0 \dots N_x-2$; the excluded last index
is the eastern wall, prescribed rather than computed. The $v$ equation is the
same construction on the second array axis.

Friction and wind are algebraic source terms rather than derivatives, so
discretizing them at level $n$ just appends them to the predictor
(`swe.py:185-192`). Coriolis *cannot* be added this way — it couples $u$ to $v$,
and an explicit treatment changes the vector's length instead of only its
direction, so it needs the corrector in §4.3.

**Continuity: a control volume rather than a difference.** The elevation
equation could be differenced term by term, but integrating it over a grid cell
instead buys exact conservation. With $h = \eta + H$,

$$
\frac{\partial h}{\partial t} = -\nabla\cdot(h\mathbf{u})
$$

integrate over the cell
$C_{ij} = [x_{i-\frac12}, x_{i+\frac12}] \times [y_{j-\frac12}, y_{j+\frac12}]$
and apply the divergence theorem:

$$
\frac{d}{dt}\iint_{C_{ij}} h \; dA
= -\oint_{\partial C_{ij}} h\,\mathbf{u}\cdot\mathbf{n} \; dl
$$

For a rectangular cell the contour integral is just four face fluxes. Dividing
by the cell area $\Delta x\,\Delta y$ and writing $\eta_{ij}$ for the cell mean
(with $H$ constant, so $\partial h/\partial t = \partial \eta/\partial t$):

$$
\frac{d \eta_{ij}}{dt}
= -\frac{F_{i+\frac12,\,j} - F_{i-\frac12,\,j}}{\Delta x}
  -\frac{G_{i,\,j+\frac12} - G_{i,\,j-\frac12}}{\Delta y},
\qquad F = hu, \quad G = hv
$$

A forward Euler step in time then gives `swe.py:224` exactly. Two details are
left over, and they are precisely what the rest of the continuity code is doing:

1. **$u$ is already on the face** where $F$ is needed — no interpolation
   required. Another consequence of the C-grid choice.
2. **$h$ is not** — it lives at centers and must be estimated on the face. The
   upwind choice (§4.4) takes it from whichever side the flow arrives from. It
   is equivalent to a centered average plus a diffusive correction,

$$
h_{\text{face}} = \underbrace{\frac{h_L + h_R}{2}}_{\text{centered}}
- \underbrace{\mathrm{sign}(u)\,\frac{h_R - h_L}{2}}_{\text{numerical diffusion}}
$$

   i.e. an implicit diffusivity of order $|u|\Delta x/2$. That drops the spatial
   accuracy of the transport term to first order, and is what keeps the
   nonlinear term stable and monotone — a centered average combined with an
   explicit time step is dispersive and unstable here.

**Why flux form conserves mass exactly.** Summing the update over all cells,
each interior face flux appears twice — once as an outflow from one cell, once
as an inflow to its neighbour — with opposite signs, so the sum telescopes down
to boundary fluxes alone. Those are zero by construction (§4.4), hence

$$
\sum_{i,j} \eta^{n+1}_{i,j} = \sum_{i,j} \eta^{n}_{i,j}
$$

to roundoff, for any $\Delta t$ — this holds regardless of stability, so a
constant `Mass:` diagnostic confirms the bookkeeping but *not* that the run is
well resolved.

### 4.2 Momentum predictor — forward in time, centered in space

$$
u^{*}_{i+\frac12,\,j}
= u^n_{i+\frac12,\,j}
- \frac{g\,\Delta t}{\Delta x}\left(\eta^n_{i+1,j} - \eta^n_{i,j}\right)
$$

$$
v^{*}_{i,\,j+\frac12}
= v^n_{i,\,j+\frac12}
- \frac{g\,\Delta t}{\Delta y}\left(\eta^n_{i,j+1} - \eta^n_{i,j}\right)
$$

(`swe.py:181-182`.) Optional terms are then added to the predictor:

$$
u^{*} \mathrel{-}= \Delta t\,\kappa\,u^n,
\qquad
u^{*} \mathrel{+}= \frac{\Delta t\,\tau_x}{\rho_0 H}
$$

and likewise for $v$ (`swe.py:185-192`).

**Sequencing.** Note that $\eta$ is still at level $n$ in both expressions,
while the elevation update (§4.4) uses the *new* velocities. That
forward–backward asymmetry is what makes the scheme stable at all; §4.5 works
through why, and derives the resulting limit on $\Delta t$.

### 4.3 Coriolis corrector — semi-implicit rotation

Rotation cannot be handled by an explicit step without amplitude error. Define

$$
\alpha_j = f_j\,\Delta t,
\qquad
\beta_c = \frac{\alpha_j^2}{4}
$$

and correct the predictor (`swe.py:196-197`):

$$
u^{n+1} = \frac{u^{*} - \beta_c\,u^n + \alpha\,v^n}{1 + \beta_c},
\qquad
v^{n+1} = \frac{v^{*} - \beta_c\,v^n - \alpha\,u^n}{1 + \beta_c}
$$

This is the trapezoidal (Crank–Nicolson) treatment of the rotation terms,

$$
u^{n+1} - u^{*} = \frac{\alpha}{2}\left(v^n + v^{n+1}\right),
\qquad
v^{n+1} - v^{*} = -\frac{\alpha}{2}\left(u^n + u^{n+1}\right)
$$

solved analytically for the 2×2 system — which is exactly where the
$1 + \alpha^2/4$ denominator comes from.

**The scheme is an exact discrete rotation.** Set the pressure gradient aside
($u^{*} = u^n$) and combine the components as $\omega = u + iv$:

$$
\omega^{n+1}
= \omega^{n}\,\frac{1 - \frac{\alpha^2}{4} - i\alpha}{1 + \frac{\alpha^2}{4}}
$$

The amplification factor has modulus

$$
\left|\frac{1 - \frac{\alpha^2}{4} - i\alpha}{1 + \frac{\alpha^2}{4}}\right|
= \frac{\sqrt{\left(1 - \frac{\alpha^2}{4}\right)^2 + \alpha^2}}
       {1 + \frac{\alpha^2}{4}}
= \frac{\sqrt{\left(1 + \frac{\alpha^2}{4}\right)^2}}{1 + \frac{\alpha^2}{4}}
= 1
$$

and phase

$$
\theta = -\arctan\!\left(\frac{\alpha}{1 - \alpha^2/4}\right)
       = -2\arctan\!\left(\frac{\alpha}{2}\right)
$$

So each step rotates the velocity vector by exactly
$2\arctan(f\Delta t/2) \approx f\Delta t$ with **no** amplitude change. The
choice $\beta_c = \alpha^2/4$ is precisely what makes the modulus unity.
Consequently the stated requirement $\alpha \ll 1$ is an *accuracy* condition
(so that $2\arctan(\alpha/2) \approx \alpha$), not a stability one. The default
configuration reports $\max\alpha \approx 2.4\times10^{-3}$, so the discrete
rotation rate is accurate to $O(\alpha^2) \sim 10^{-6}$.

Boundary conditions are then reimposed (`swe.py:199-200`):

$$
u^{n+1}_{N_x-\frac12,\,j} = 0,
\qquad
v^{n+1}_{i,\,N_y-\frac12} = 0
$$

The western and southern walls are enforced structurally, by omitting their
face fluxes from the divergence (§4.4).

### 4.4 Continuity — upwind fluxes

The layer thickness at each cell face is taken from the upstream side
(`swe.py:204-214`):

$$
h^{E}_{i+\frac12,\,j} =
\begin{cases}
H + \eta^n_{i,j},   & u^{n+1}_{i+\frac12,\,j} > 0\\[4pt]
H + \eta^n_{i+1,j}, & \text{otherwise}
\end{cases}
\qquad
h^{W}_{i-\frac12,\,j} = h^{E}_{i-\frac12,\,j}
$$

$$
h^{N}_{i,\,j+\frac12} =
\begin{cases}
H + \eta^n_{i,j},   & v^{n+1}_{i,\,j+\frac12} > 0\\[4pt]
H + \eta^n_{i,j+1}, & \text{otherwise}
\end{cases}
\qquad
h^{S}_{i,\,j-\frac12} = h^{N}_{i,\,j-\frac12}
$$

The face mass fluxes and their divergence (`swe.py:216-220`) are

$$
F_{i+\frac12,\,j} = u^{n+1}_{i+\frac12,\,j}\, h^{E}_{i+\frac12,\,j},
\qquad
G_{i,\,j+\frac12} = v^{n+1}_{i,\,j+\frac12}\, h^{N}_{i,\,j+\frac12}
$$

$$
\texttt{uhwe}_{i,j} = F_{i+\frac12,\,j} - F_{i-\frac12,\,j},
\qquad
\texttt{vhns}_{i,j} = G_{i,\,j+\frac12} - G_{i,\,j-\frac12}
$$

giving the elevation update (`swe.py:224`)

$$
\eta^{n+1}_{i,j} = \eta^n_{i,j}
- \Delta t \left(
\frac{F_{i+\frac12,\,j} - F_{i-\frac12,\,j}}{\Delta x}
+ \frac{G_{i,\,j+\frac12} - G_{i,\,j-\frac12}}{\Delta y}
\right)
$$

plus the optional source/sink terms (`swe.py:227-232`):

$$
\eta^{n+1} \mathrel{+}= \Delta t\,\sigma,
\qquad
\eta^{n+1} \mathrel{-}= \Delta t\,w
$$

At $i = 0$ the code uses $\texttt{uhwe}_{0,j} = F_{\frac12,\,j}$, i.e. the
west-face flux is structurally zero — that *is* the western wall. Same for
$j = 0$ and the southern wall, and $u^{n+1} = 0$ closes the east. Because the
scheme is written in flux-divergence form with zero fluxes on every boundary,

$$
\sum_{i,j} \eta^{n+1}_{i,j} = \sum_{i,j} \eta^{n}_{i,j}
$$

exactly (to roundoff). This is what the `Mass:` diagnostic printed during the
run is checking.

Upwinding the thickness is what keeps the nonlinear transport stable; the price
is first-order accuracy and some implicit numerical diffusion.

---

### 4.5 Stability: why forward–backward, and where $\Delta t$ comes from

The ordering of the updates is not cosmetic. Take the linearized, non-rotating,
one-dimensional pair as implemented — momentum from $\eta^n$, then elevation
from the *new* $u^{n+1}$:

$$
u^{n+1}_{i+\frac12} = u^{n}_{i+\frac12}
- \frac{g\Delta t}{\Delta x}\left(\eta^n_{i+1} - \eta^n_{i}\right)
$$

$$
\eta^{n+1}_{i} = \eta^{n}_{i}
- \frac{H\Delta t}{\Delta x}\left(u^{n+1}_{i+\frac12} - u^{n+1}_{i-\frac12}\right)
$$

Insert a Fourier mode $\eta^n_i = \hat{\eta}\,\lambda^n e^{ikx_i}$,
$u^n_{i+\frac12} = \hat{u}\,\lambda^n e^{ik(x_i + \Delta x/2)}$. With
$e^{ik\Delta x/2} - e^{-ik\Delta x/2} = 2i\sin(k\Delta x/2)$, and writing

$$
p = \sqrt{gH}\,\Delta t\,\frac{2\sin(k\Delta x/2)}{\Delta x}
$$

the two lines become

$$
\lambda \hat{u} = \hat{u} - i a \hat{\eta},
\qquad
\lambda \hat{\eta} = (1 - p^2)\,\hat{\eta} - i b \hat{u},
\qquad ab = p^2
$$

where $a = g\Delta t\,\sigma/\Delta x$, $b = H\Delta t\,\sigma/\Delta x$ and
$\sigma = 2\sin(k\Delta x/2)$. The $-p^2$ term — and the whole result — comes
from the substitution of $u^{n+1}$ into the second equation. The amplification
matrix

$$
M = \begin{pmatrix} 1 & -ia \\ -ib & 1 - p^2 \end{pmatrix}
$$

has

$$
\operatorname{tr} M = 2 - p^2,
\qquad
\det M = (1 - p^2) - (-ia)(-ib) = 1 - p^2 + ab = 1
$$

**The determinant is exactly 1.** Since $\lambda_1\lambda_2 = \det M = 1$, the
eigenvalues satisfy

$$
\lambda^2 - (2 - p^2)\lambda + 1 = 0
$$

If $|2 - p^2| \le 2$ the roots are a complex-conjugate pair on the unit circle,
$|\lambda_1| = |\lambda_2| = 1$: the mode neither grows nor decays, it only
advances in phase. Outside that range the roots are real and reciprocal, so one
of them exceeds 1 and the mode blows up. The condition is therefore

$$
p \le 2
$$

The worst case is the shortest resolvable wave, $k\Delta x = \pi$, where
$\sigma = 2$ and $p = 2\sqrt{gH}\,\Delta t/\Delta x$. That gives

$$
\Delta t \le \frac{\Delta x}{\sqrt{gH}}
$$

which is the CFL condition quoted in the module docstring: a signal travelling
at $\sqrt{gH}$ must not cross a grid cell in one step.

**Contrast with a simultaneous explicit step.** Had the elevation been updated
from $u^{n}$ instead of $u^{n+1}$, the same analysis gives
$\det M = 1 + p^2$ and $\lambda = 1 \pm ip$, so

$$
|\lambda| = \sqrt{1 + p^2} > 1 \quad \text{for every } \Delta t > 0
$$

Unconditionally unstable — no choice of time step helps. Reusing the freshly
computed velocity, at no extra cost, is the entire difference between a scheme
that works and one that cannot.

**In two dimensions** both directions contribute to $p^2$:

$$
p^2 = gH\,\Delta t^2\left(
\frac{4\sin^2(k\Delta x/2)}{\Delta x^2} + \frac{4\sin^2(l\Delta y/2)}{\Delta y^2}
\right)
$$

so $p \le 2$ tightens to

$$
\Delta t \le \frac{1}{\sqrt{gH}}
\left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}\right)^{-1/2}
\;\xrightarrow{\ \Delta x = \Delta y\ }\;
\frac{\Delta x}{\sqrt{2\,gH}}
$$

a factor $\sqrt{2}$ stricter than the 1D form in the docstring. The code sets
$\Delta t = 0.1\min(\Delta x,\Delta y)/\sqrt{gH}$ (`swe.py:59`), giving
$\Delta t = 21.4$ s against a 2D limit of 151.5 s — about 14% of the limit.

That margin is deliberate, since the analysis above omits several things the
real run includes: the nonlinear term makes the effective wave speed
$\sqrt{g(H+\eta)}$ rather than $\sqrt{gH}$; boundary reflections superpose
modes; and the analysis is linear while the transport term is not. Two effects
work in the safe direction — friction is purely damping, and the Coriolis
corrector is exactly neutral (§4.3), so rotation contributes no growth of its
own.

---

## 5. Derived scales reported at startup

Computed at `swe.py:97-110` and written to `param_output.txt`:

$$
\text{Rossby deformation radius:} \quad L_R = \frac{\sqrt{gH}}{f_0}
$$

$$
\text{Rossby number:} \quad \mathrm{Ro} = \frac{\sqrt{gH}}{f_0 L_x}
$$

$$
\text{Long Rossby wave speed:} \quad c_R = \frac{\beta g H}{f_0^2}
$$

$$
\text{Rossby transit time:} \quad T_R = \frac{L_x}{c_R}
$$

$$
\text{Gravity wave speed:} \quad c = \sqrt{gH}
$$

For the default configuration ($g = 9.81$, $H = 100$, $f_0 = 10^{-4}$,
$\beta = 2\times10^{-11}$, $L_x = L_y = 10^6$): $c \approx 31.3\ \mathrm{m/s}$,
$L_R \approx 313\ \mathrm{km}$, $c_R \approx 1.96\ \mathrm{m/s}$,
$T_R \approx 5.9\ \mathrm{days}$, $\Delta t = 21.4\ \mathrm{s}$.

---

## 6. Code layout

| File | Role |
|---|---|
| `swe.py` | parameters, initial conditions, main time loop |
| `viz_tools.py` | animations and diagnostic plots |
| `fourier_transform.py` | hand-rolled DFT used by the spectrum plot |
| `scenarios.txt` | parameter sets for Kelvin- and Rossby-wave regimes |
| `param_output.txt` | generated: parameters of the last run |

### Time-loop structure (`swe.py:179-254`)

1. momentum predictor $u^{*}, v^{*}$ — pressure gradient, then friction/wind
2. Coriolis corrector → $u^{n+1}, v^{n+1}$
3. reimpose $u = 0$ (east), $v = 0$ (north)
4. upwind face thicknesses and flux divergences
5. $\eta^{n+1}$, then source/sink
6. copy $n{+}1 \to n$ (`np.copy`, which is also what makes the stored animation
   frames independent arrays)
7. sample diagnostics

### Sampling

- `anim_interval = 20` — store full $\eta, u, v$ fields for the animations
- `sample_interval = 1000` — store a mid-domain $x$-section (Hovmöller) and the
  center-point value (time series / spectrum)

### Visualization (`viz_tools.py`)

| Function | Output |
|---|---|
| `eta_animation` | `pcolormesh` movie of surface elevation |
| `velocity_animation` | quiver movie of $(u,v)$ |
| `eta_animation3D` | 3D surface movie of $\eta$ — this is what `surface.gif` in the README shows |
| `pmesh_plot`, `quiver_plot` | single-frame snapshots |
| `hovmuller_plot` | $x$–$t$ diagram; crest slope *is* the phase speed |
| `plot_time_series_and_ft` | center-point $\eta(t)$ and its power spectrum |

The animation color scale is fixed from the middle frame
(`viz_tools.py:22-23`), so it needs adjusting if parameters change
substantially. The last two diagnostics are the most useful for identifying
wave types and are commented out at `swe.py:263-266`.

### Spectrum (`fourier_transform.py`)

A direct $O(N^2)$ discrete Fourier transform of a signal $s_q$,
$q = 1 \dots N$, over record length $T$:

$$
A_n = \frac{2}{N}\sum_{q=1}^{N} s_q \cos\!\left(\frac{2\pi q n}{N}\right),
\qquad
B_n = \frac{2}{N}\sum_{q=1}^{N} s_q \sin\!\left(\frac{2\pi q n}{N}\right)
$$

with the Nyquist term $A_{N/2} = \frac{1}{N}\sum_q s_q \cos(\pi q)$,
$B_{N/2} = 0$, returning the power spectrum and its frequency axis

$$
E_n = A_n^2 + B_n^2,
\qquad
\nu_n = \frac{n}{T}
$$

---

## 7. Configuration switches (`swe.py:42-47`)

| Flag | Effect |
|---|---|
| `use_coriolis` | enable $f$; without it, pure gravity waves |
| `use_beta` | $f = f_0 + \beta y$ instead of $f = f_0$; required for Rossby waves |
| `use_friction` | linear bottom drag, $\kappa_0 = 1/(5\ \text{days})$ |
| `use_wind` | steady surface stress |
| `use_source` / `use_sink` | mass injection / removal |

`scenarios.txt` gives tuned parameter sets. The Rossby-wave scenario uses a
reduced gravity $g = 0.01$ to slow the waves into the quasi-geostrophic regime.

---

## 8. Fixed issues

All of the following were latent in the original code — harmless in the default
configuration, but triggered as soon as the corresponding switch was flipped.
They have been fixed; recorded here because the failure modes are instructive
and easy to reintroduce.

1. **Friction on $v$ used the wrong slice.** `swe.py` applied
   `v_np1[:-1, :] -= dt*kappa[:-1, :]*v_n[:-1, :]`, a copy of the $u$ line, but
   $v$'s valid domain is `[:, :-1]` (north faces), not `[:-1, :]`. The effect was
   that the easternmost column of $v$ received no bottom drag at all, while the
   north-wall row was damped pointlessly before being zeroed by the boundary
   condition. Now mirrors the slicing of the $v$ predictor.
2. **Wind stress was disabled and shape-fragile.** `tau_x` carried a trailing
   `*0`, so `use_wind = True` applied no zonal stress; `tau_y` was shaped
   `(1, N_x)` and broadcast against an array whose last axis is $N_y$, which
   worked only for a square grid and raised `ValueError` otherwise. The $v$ wind
   term also had the same slicing error as (1). `tau_y` is now a length-$N_y$
   array, sliced to the $v$ domain.
3. **`use_coriolis` without `use_beta` crashed.** `L_R` and `c_R` were bound only
   inside the `use_beta` branch, but the diagnostics printed them
   unconditionally, so a constant-$f$ run died with
   `NameError: name 'L_R' is not defined`. $L_R = \sqrt{gH}/f_0$ does not depend
   on $\beta$ and is now computed unconditionally; the two long-Rossby-wave
   diagnostics, which genuinely require a $\beta$-plane, are printed only when
   `use_beta` is on.
4. **`use_sink` without `use_source` crashed obscurely.** The sink is defined as
   the uniform rate that removes exactly as much mass as the source adds, so it
   is meaningless alone — but it failed with a bare `NameError` on `sigma`. It
   now raises an explicit `ValueError` naming the requirement.
5. **Dead line in the DFT.** `fourier_transform.py` set `A[0] = signal.mean()`,
   which the $n = 1$ loop iteration immediately overwrote — slot $n-1$ holds
   harmonic $n$, so index 0 is the fundamental, not DC. Removed, with a comment
   noting that the returned spectrum starts at harmonic 1 and excludes DC by
   design. (To remove a mean offset properly, detrend the signal *before*
   transforming.)
6. **Mislabelled diagnostic.** The startup banner printed
   $\sqrt{gH}/(f_0 L_x)$ as "Rossby number". That expression is $L_R/L_x$ — the
   deformation radius as a fraction of the domain, i.e. the square root of the
   Burger number. The Rossby number proper is $U/fL$. The value is unchanged;
   only the label now says what it is.
7. **Docstring and comment typos** — the second continuity flux read
   $\partial[(\eta+H)u]/\partial y$ instead of $v$; plus "mst", "eqations",
   "solves that solves", "enxt", "anin_interval", and a stray `km` unit on the
   printed $\rho_0$.
8. **`viz_tools.py:30`** used the pre-matplotlib-3.3 `set_array` convention
   (`eta_list[num][:-1, :-1].flatten()`), which raises `ValueError` on modern
   matplotlib since $X$, $Y$ and $\eta$ all share the same shape and
   `shading="nearest"` is selected. Now passes the full 2D array.

**Verification.** With the default switches the fixes are a no-op: `eta_n`,
`u_n` and `v_n` are bit-identical to the pre-fix code after 400 steps. Each
previously-broken path — friction, wind, constant $f$, source+sink, and a
non-square 100×150 grid — now runs to completion with finite fields and mass
conserved to the initial value.

One item was deliberately **not** changed: the friction switch damps momentum
but the model has no explicit diffusion of $\eta$, so the only smoothing of the
elevation field is the implicit numerical diffusion of the upwind scheme
(§4.1). That is a modelling choice, not a defect.

---

## 9. Running it

The environment on this host has no `pip` for the system Python and no system
`ffmpeg`, so the repo carries a local setup:

```bash
python3 -m venv --system-site-packages .venv   # inherits the RPM numpy
.venv/bin/python -m pip install matplotlib imageio-ffmpeg
.venv/bin/python swe.py
```

`matplotlibrc` in the repo root points `animation.ffmpeg_path` at the static
ffmpeg binary shipped inside the `imageio-ffmpeg` wheel, so it is only picked up
when running from this directory. All three animation functions take
`filetype = "mp4"` (ffmpeg) or `"gif"` (pillow, no external binary needed).
As configured, output is `surface.mp4` and `velocity.mp4`, matching the pair of
gifs in the README; uncomment the `eta_animation` call in `swe.py` to also get
the flat `pcolormesh` version. The
default 5000-step run takes a few seconds to integrate and rather longer to
encode.
