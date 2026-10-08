# DynamiX: the mathematics behind the engine

This document describes what the NumPy backend (`src/dynamix/numpy_backend/`) computes, and why.
Symbols map directly to code; file references are given in each section.

## 1. Conventions

Coordinates and frames:

- The world frame is right-handed with the Z axis pointing up. Gravity is $\mathbf{g} = (0, 0, -9.81)$ m/s².
- Body $i$ has a centre-of-mass position $\mathbf{x}_i \in \mathbb{R}^3$ and a unit quaternion
  $\mathbf{p}_i = [x, y, z, w]$ (scalar part last) with rotation matrix $R_i$.
- Together, $q_i = (\mathbf{x}_i, \mathbf{p}_i) \in \mathbb{R}^7$. These are maximal coordinates: six degrees of freedom per body.

Velocities:

- The generalized velocity is $u_i = (\mathbf{v}_i, \boldsymbol{\omega}_i) \in \mathbb{R}^6$.
- The linear velocity $\mathbf{v}_i$ is expressed in the world frame.
- The angular velocity $\boldsymbol{\omega}_i$ is expressed in the body frame.

Inertia and notation:

- Body $i$ has mass $m_i$ and a constant body-frame inertia tensor $I_i$ about its centre of mass.
- For $\mathbf{a} \in \mathbb{R}^3$, the matrix $[\mathbf{a}]_\times$ is the skew matrix with $[\mathbf{a}]_\times \mathbf{b} = \mathbf{a} \times \mathbf{b}$.

The state of all $n$ bodies is stored as arrays `q` of shape `(n, 7)` and `u` of shape `(n, 6)`
(see [buffers.py](../src/dynamix/core/buffers.py)).

## 2. Kinematics

The position obeys $\dot{\mathbf{x}}_i = \mathbf{v}_i$. For a body-frame angular velocity, the orientation obeys

$$
\dot{R}_i = R_i \, [\boldsymbol{\omega}_i]_\times ,
\qquad
\dot{\mathbf{p}}_i = \frac{1}{2} \, \mathbf{p}_i \otimes (\boldsymbol{\omega}_i, 0) ,
$$

where $\otimes$ is the Hamilton product. For a constant $\boldsymbol{\omega}$ over a time $h$, the exact solution is

$$
\mathbf{p}(t+h) = \mathbf{p}(t) \otimes \exp(h\,\boldsymbol{\omega}) ,
$$

with the quaternion exponential of a rotation vector $\boldsymbol{\phi}$ given by

$$
\exp(\boldsymbol{\phi}) =
\left(
\frac{\boldsymbol{\phi}}{\lVert \boldsymbol{\phi} \rVert}
\sin\frac{\lVert \boldsymbol{\phi} \rVert}{2} ,
\;
\cos\frac{\lVert \boldsymbol{\phi} \rVert}{2}
\right) .
$$

The function `quaternion.integrate` in [quaternion.py](../src/dynamix/numpy_backend/quaternion.py)
implements this. It uses a series expansion of the sine term near zero and renormalizes the result.
The quaternion therefore stays on the unit sphere, and orientation is never integrated component-wise.
Because $\boldsymbol{\omega}$ is in the body frame, the update multiplies on the right.

## 3. Equations of motion

With a body-frame angular velocity, the Newton–Euler equations of one body are

$$
m \, \dot{\mathbf{v}} = m \, \mathbf{g} + \mathbf{f}_c ,
\qquad
I \, \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times I \boldsymbol{\omega} = \boldsymbol{\tau}_c .
$$

Written for all bodies together,

$$
M \, \dot{u} = h(u) + W \boldsymbol{\lambda} ,
$$

with the block-diagonal mass matrix and the smooth force vector

$$
M = \operatorname{diag}\left( m_1 \mathbf{1}_3, \, I_1, \, \dots, \, m_n \mathbf{1}_3, \, I_n \right) ,
\qquad
h = \begin{pmatrix} m \, \mathbf{g} \\ -\boldsymbol{\omega} \times I \boldsymbol{\omega} \end{pmatrix} .
$$

Here $\mathbf{1}_3$ is the $3 \times 3$ identity, and $W \boldsymbol{\lambda}$ collects constraint forces and torques (Section 4).

The advantage of a body-frame angular velocity is that $M$ is constant in time. The inverse inertias are
computed once in `Engine.__init__`, instead of rotating $I$ into the world frame in every step.

## 4. Bilateral constraints

### 4.1 Ball joint

A ball (spherical) joint between bodies $a$ and $b$ requires two body-fixed anchor points to coincide.
With the anchors $\mathbf{r}_a$ and $\mathbf{r}_b$ given in the respective body frames, the constraint is

$$
\mathbf{g}(q) = \left( \mathbf{x}_b + R_b \mathbf{r}_b \right) - \left( \mathbf{x}_a + R_a \mathbf{r}_a \right) = \mathbf{0} .
$$

Body $a$ may be the world. In that case $R_a$ is the identity, $\mathbf{x}_a = \mathbf{0}$, and $\mathbf{r}_a$ is a
fixed point in the world frame. Each joint removes three degrees of freedom. The residual is computed by
`ball_joint_residual` in [constraints.py](../src/dynamix/numpy_backend/constraints.py).

### 4.1b Hinge joint

A hinge adds two rotational rows to the three point rows of the ball joint.
Let $n_a, n_b$ be the hinge axis in body frames a and b, and let $p_1, p_2$ be
two unit vectors in frame a perpendicular to $n_a$. The extra residuals are

$$
g_{3+i} = (R_a p_i)^\top (R_b n_b), \qquad i = 1, 2,
$$

which vanish exactly when the world axes $R_a n_a$ and $R_b n_b$ coincide.
Their body-frame angular Jacobian rows are

$$
\frac{\partial g_{3+i}}{\partial \omega_a} = p_i \times \left(R_a^\top R_b n_b\right),
\qquad
\frac{\partial g_{3+i}}{\partial \omega_b} = n_b \times \left(R_b^\top R_a p_i\right),
$$

and the linear parts are zero. A hinge therefore has 5 rows, a ball joint 3.

In a mixed system all joints are padded to the largest row count. Padded rows
have a zero Jacobian and a unit diagonal in $G$, so their multipliers are zero
and the banded solve keeps a uniform block size. The initial configuration must
satisfy $R_a n_a = R_b n_b$.

### 4.1c Fixed joint

A fixed joint keeps the anchors together (the three point rows of the ball joint) and the
relative orientation at its initial value $C = R_a^\top R_b\big|_{t=0}$. With

$$
M = C^\top R_a^\top R_b,
$$

which is the identity at rest, the three rotation residuals and their Jacobian blocks are

$$
g_{\mathrm{rot}} = \operatorname{vee}\left(\tfrac12 (M - M^\top)\right),
\qquad
\frac{\partial g_{\mathrm{rot}}}{\partial \omega_b} = \tfrac12\left(\operatorname{tr}(M) I - M^\top\right),
\qquad
\frac{\partial g_{\mathrm{rot}}}{\partial \omega_a} = -\tfrac12\left(\operatorname{tr}(M) I - M\right) C^\top .
$$

They follow from $\dot M = -[C^\top \omega_a]_\times M + M [\omega_b]_\times$ and the identity
$\operatorname{vee}\left(\operatorname{skew}(M[w]_\times)\right) = \tfrac12(\operatorname{tr}(M) I - M^\top) w$.
A fixed joint has 6 rows.

### 4.1d Prismatic joint

A prismatic joint keeps the relative orientation fixed (the three rows above) and restricts the
displacement

$$
d = (x_b + R_b r_b) - (x_a + R_a r_a)
$$

to the slide axis $n_a$ given in frame a. With $p_1, p_2$ two unit vectors perpendicular to $n_a$,
the two translation rows are

$$
g_i = (R_a p_i)^\top d, \qquad i = 1, 2 .
$$

Because the direction $R_a p_i$ rotates with body a, the linear Jacobian is no longer the constant
$E$ of the other joints:

$$
\frac{\partial g_i}{\partial v_b} = -\frac{\partial g_i}{\partial v_a} = (R_a p_i)^\top,
\qquad
\frac{\partial g_i}{\partial \omega_a} = p_i \times \left(r_a + R_a^\top d\right),
\qquad
\frac{\partial g_i}{\partial \omega_b} = r_b \times \left(R_b^\top R_a p_i\right).
$$

A system containing a prismatic joint therefore carries the linear blocks explicitly, and the
Delassus blocks become $\frac{1}{m} L_s L_t^\top + A_s I^{-1} A_t^\top$ with $L$ the linear and
$A$ the angular block. The initial displacement must lie along the axis.

### 4.1e Joint numbering

The Delassus matrix is banded only if joints sharing a body have nearby indices. Joints are
renumbered internally by reverse Cuthill-McKee on the graph of joints sharing a body, but only if
that strictly narrows the band; results are independent of the creation order.

### 4.2 Velocity-level form

DynamiX enforces constraints on velocities. Differentiating $\mathbf{g} = \mathbf{0}$ in time gives

$$
\dot{\mathbf{g}} = J \, u = \mathbf{0} ,
\qquad
J = \frac{\partial \mathbf{g}}{\partial u} .
$$

With $\dot{\mathbf{x}} = \mathbf{v}$ and $\dot{R} \, \mathbf{r} = R \, [\boldsymbol{\omega}]_\times \mathbf{r} = - R \, [\mathbf{r}]_\times \boldsymbol{\omega}$,
the Jacobian blocks of one joint are

$$
J_b = \begin{pmatrix} \mathbf{1}_3 & -R_b [\mathbf{r}_b]_\times \end{pmatrix} ,
\qquad
J_a = \begin{pmatrix} -\mathbf{1}_3 & R_a [\mathbf{r}_a]_\times \end{pmatrix} .
$$

Each block is placed in the columns of its body, so one joint contributes a $3 \times 6n$ row block.
The block of the world body is omitted. The function `ball_joint_jacobian` builds it.

The analytical Jacobian is verified against central finite differences of $\mathbf{g}$ in
`tests/test_constraint_jacobian.py`. The orientation is perturbed with the exponential map, so the
perturbation matches the velocity parametrization.

### 4.3 Constraint forces

Constraint forces do no work along admissible motions, so they lie in the range of $J^\top$. Hence

$$
W = J^\top ,
$$

and $\boldsymbol{\lambda} \in \mathbb{R}^{3 n_j}$ are the Lagrange multipliers of the $n_j$ joints.

## 5. Time stepping: Moreau's theta method

Let $\Delta t$ be the step size and $\theta \in (0, 1]$ the method parameter (default $\theta = 1/2$, the midpoint rule).
Starting from $(q_k, u_k)$, one step in [engine.py](../src/dynamix/numpy_backend/engine.py) has four stages.

**Stage 1: midpoint configuration.**

$$
\mathbf{x}_\theta = \mathbf{x}_k + \theta \, \Delta t \, \mathbf{v}_k ,
\qquad
\mathbf{p}_\theta = \mathbf{p}_k \otimes \exp\left( \theta \, \Delta t \, \boldsymbol{\omega}_k \right) .
$$

**Stage 2: velocity update.** Momentum balance with constraint impulses, with the Jacobian evaluated at $q_\theta$:

$$
M \left( u_{k+1} - u_k \right) = \Delta t \, h(u_k) + J(q_\theta)^\top \boldsymbol{\Lambda} ,
\qquad
\boldsymbol{\Lambda} = \Delta t \, \boldsymbol{\lambda} .
$$

**Stage 3: velocity-level constraint** at $q_\theta$:

$$
J(q_\theta) \, u_{k+1} = - \frac{\gamma}{\Delta t} \, \mathbf{g}(q_\theta) ,
$$

where $\gamma$ is the `stabilization` parameter (Section 6).

**Stage 4: position update** over the remaining part of the step, using the new velocity:

$$
\mathbf{x}_{k+1} = \mathbf{x}_\theta + (1-\theta) \, \Delta t \, \mathbf{v}_{k+1} ,
\qquad
\mathbf{p}_{k+1} = \mathbf{p}_\theta \otimes \exp\left( (1-\theta) \, \Delta t \, \boldsymbol{\omega}_{k+1} \right) .
$$

Evaluating $J$ and $\mathbf{g}$ at the midpoint keeps a structure that carries over unchanged to unilateral
(contact) constraints, where velocity-level impulses are the natural unknowns. At this stage only bilateral
constraints are implemented.

The gravity and gyroscopic terms $h(u_k)$ are treated explicitly, so the velocity update is first-order
accurate for general three-dimensional rotation. This is validated against a reference solution in
`tests/test_chain.py`, including a convergence check in $\Delta t$.

### 5.1 Solving for the multipliers

Define the unconstrained ("free") velocity

$$
u^\ast = u_k + \Delta t \, M^{-1} h(u_k) .
$$

Substituting Stage 2 into Stage 3 gives the Schur-complement (Delassus) system

$$
G \, \boldsymbol{\Lambda} = - \frac{\gamma}{\Delta t} \, \mathbf{g}(q_\theta) - J \, u^\ast ,
\qquad
G = J M^{-1} J^\top ,
$$

and the new velocity follows as

$$
u_{k+1} = u^\ast + M^{-1} J^\top \boldsymbol{\Lambda} .
$$

The matrix $G \in \mathbb{R}^{3 n_j \times 3 n_j}$ is symmetric positive definite whenever the constraints are
independent.

**Block-sparse Jacobian.** A joint touches at most two bodies, so $J$ is never formed as a dense
$3 n_j \times 6 n$ matrix. Each joint has one *side* per attached body (the world has none). For the side
of body $i$ with sign $s = +1$ for body $b$ and $s = -1$ for body $a$, the Jacobian block is

$$
J_{\text{side}} = \begin{pmatrix} s \, \mathbf{1}_3 & -s \, R_i [\mathbf{r}]_\times \end{pmatrix} ,
$$

and only the $3 \times 3$ rotational part changes from step to step. The products $J u$ and
$M^{-1} J^\top \boldsymbol{\Lambda}$ are evaluated side by side and scattered to joints or bodies.

**Banded Delassus matrix.** The $3 \times 3$ block $(j, k)$ of $G$ is non-zero only when joints $j$ and $k$
share a body, and it is the sum over the shared bodies $i$ of

$$
G_{jk} = \sum_i J_{j,i} \, M_i^{-1} \, J_{k,i}^\top ,
\qquad
M_i^{-1} = \operatorname{diag}\left( \frac{1}{m_i} \mathbf{1}_3, \, I_i^{-1} \right) .
$$

The list of such side pairs is built once from the topology. Per step, the blocks are computed in one batched
product and scattered into the lower triangle of a banded matrix. For a chain with joints numbered along the
chain, $G$ is block tridiagonal with a constant half-bandwidth of 5 scalar entries, independent of $n$.

**Banded Cholesky solve.** $G \boldsymbol{\Lambda} = \mathbf{b}$ is solved with a banded Cholesky factorization
(`scipy.linalg.solveh_banded`, LAPACK `pbsv`) at a cost of $O(n_j \, k_d^2)$ for half-bandwidth $k_d$, so the
cost per step is linear in the chain length. If SciPy is not installed, the band is expanded and solved densely
with `numpy.linalg.solve`; the result is the same, only slower. For other topologies the bandwidth follows the
joint numbering: joints that share a body should have nearby indices, otherwise the band becomes wide.

## 6. Drift stabilization

Enforcing only $\dot{\mathbf{g}} = \mathbf{0}$ lets $\mathbf{g}$ drift because of integration error. The right-hand side
of Stage 3 therefore contains a Baumgarte-type correction

$$
\dot{\mathbf{g}} = - \frac{\gamma}{\Delta t} \, \mathbf{g} ,
$$

which reduces a non-zero $\mathbf{g}$ by roughly a factor $(1 - \gamma)$ per step. It is a numerical correction on the
velocity level, not a position-level constraint solve. The default is $\gamma = 0.2$, and $\gamma = 0$ disables it.

## 7. Diagnostics

The function `engine.total_energy` returns the sum of kinetic and gravitational potential energy:

$$
E = \sum_i \left(
\frac{1}{2} m_i \lVert \mathbf{v}_i \rVert^2
+ \frac{1}{2} \boldsymbol{\omega}_i^\top I_i \, \boldsymbol{\omega}_i
- m_i \, \mathbf{g}^\top \mathbf{x}_i
\right) .
$$

For workless constraints and an ideal integrator, $E$ is conserved. The tests check that the drift stays small
and that the joint gaps stay near zero. They also check that a torque-free asymmetric body conserves its
world-frame angular momentum $R \, I \boldsymbol{\omega}$ and its kinetic energy.

## 8. Validation reference

`tests/reference.py` integrates an independent Lagrangian model of a planar chain of $n$ uniform rods with
lengths $\ell_k$ and masses $m_k$. It uses the absolute angles $\varphi_k$ measured from the downward vertical.

Let $c_{kj}$ be the sensitivity of the centre of mass of rod $k$ to the angle $\varphi_j$:

$$
c_{kj} =
\begin{cases}
\ell_j & j < k , \\
\ell_k / 2 & j = k , \\
0 & j > k .
\end{cases}
$$

With the abbreviations

$$
A_{ij} = \sum_k m_k \, c_{ki} \, c_{kj} ,
\qquad
s_i = \sum_k m_k \, c_{ki} ,
$$

the equations of motion are

$$
\sum_j \left[ A_{ij} \cos(\varphi_i - \varphi_j) + \delta_{ij} \frac{m_i \ell_i^2}{12} \right] \ddot{\varphi}_j
= - \sum_j A_{ij} \sin(\varphi_i - \varphi_j) \, \dot{\varphi}_j^{\,2} - g \, s_i \sin\varphi_i .
$$

The reference uses minimal coordinates and a high-accuracy ODE solver, so agreement with the maximal-coordinate
engine is a meaningful check of the constraint formulation and the time stepping.

## 9. What is not covered yet

- Friction and shapes other than spheres and planes.
- A Warp backend: it will follow the same equations with its own array types.

## 10. Contacts

Contacts are frictionless, between spheres and between a sphere and a static plane. Detection is
separate from the solver: it fills a fixed-capacity buffer of contact point, normal, gap, the two
body indices and a restitution coefficient, and the solver reads nothing else.

### 10.1 SDF contact

A contact pairs a shape a with a probe sphere b of radius $r_b$ at $x_b$. With the signed distance
$\mathrm{sdf}_a$ of the shape (negative inside),

$$
\phi = \mathrm{sdf}_a(x_b) - r_b, \qquad n = \nabla \mathrm{sdf}_a(x_b),
$$

where $n$ points from a to b. For a sphere a, $\mathrm{sdf}_a(x) = |x - x_a| - r_a$; for the plane
$n_p \cdot x = d$ (world body, index $-1$), $\mathrm{sdf}(x) = n_p \cdot x - d$. The contact is
active when $\phi \le 0$ at the midpoint configuration $q_\theta$ of the step, and its point is
$p = x_b - (r_b + \phi/2)\, n$. There are no speculative contacts, so restitution is not blunted.

### 10.2 Contact law

The normal row has the linear blocks $\mp n^T$ for a and b. The angular blocks are
$\mp (R^T (r \times n))^T$ with $r = p - x$; for spheres $r \times n = 0$, so they vanish and only
linear velocities enter. With $u_N = J_N u$ the law at the velocity level is

$$
0 \le \lambda \;\perp\; J_N u^+ - b \ge 0, \qquad b = \max\!\left(-e\, u_N^-,\; -\beta \phi / \Delta t\right).
$$

Here $u_N^-$ is the normal relative velocity at the start of the step, $e$ is Newton's coefficient
(the minimum of the two bodies; applied only above an approach speed threshold) and $\beta$ is the
optional penetration recovery gain (`contact_stabilization`, default 0).

### 10.3 Block iteration

Each iteration performs a projected Jacobi update over all contacts in parallel,

$$
\lambda \leftarrow \max\!\left(0,\; \lambda - \frac{\omega\,(J_N u - b)}{s\, w}\right),
$$

with $w = 1/m_a + 1/m_b$ and $s$ the largest number of contacts on either body (this keeps the
parallel update convergent). Then the joint constraints are re-imposed with the factorised joint
Delassus matrix, $G\, \delta\Lambda = -(\gamma/\Delta t)\, g - J u$, so the final velocities satisfy
the joints exactly. The loop stops when the impulse changes are below `contact_tolerance` or after
`contact_iterations`. Without contacts the step is the joint-only path of Section 5.

### 10.4 Limitations

- Penetration of order $\Delta t\,|v|$ after a fast impact (set `contact_stabilization` to recover it).
- Jacobi converges slowly in deep piles, so the iteration cap is reached and penetration grows with
  the pile height; graph-coloured Gauss-Seidel would help.
- Tunnelling is possible if a ball moves farther than its diameter in one step.
- If more contacts are found than `max_contacts`, the deepest are kept, `overflow` is set and a
  warning is issued.

## 11. JAX backend

`dynamix.jax_backend` implements Sections 2–6 unchanged; only the data layout differs. The step is a pure
function of the state and constant parameters, so it can be jitted, batched with `vmap` and differentiated.
The Delassus matrix is stored as 3 by 3 blocks in lower block-banded form, and for more than 10 joints it is
factorized by a block Cholesky loop with closed-form 3 by 3 kernels. Small systems use a dense solve, which
is faster there.

### 11.1 Contacts in the JAX backend

Detection and solver follow Section 10 with static shapes (`dynamix.jax_backend.contacts`):

- The spheres are sorted by grid cell (edge: the largest diameter). Each sphere is tested against the next
  `slots` spheres of its own cell and of 13 forward neighbour cells (`Engine(buffers, contact_slots=4)`), so
  the candidate set has fixed size $14\,\mathrm{slots}\,n$. A cheap squared-distance test selects the hits, they
  are compacted into the fixed-capacity buffer by a prefix sum, and the SDF geometry is evaluated on the
  selected contacts only.
- `overflow` is set if a grid cell holds more than `slots` spheres or if there are more hits than
  `max_contacts`; in the latter case the deepest contacts are kept, as in the NumPy backend.
- The block iteration runs in a `lax.while_loop` with the same early exit. Entries of the buffer beyond
  `count` have zero weight. Without joints only the linear velocities are iterated; with joints each sweep is
  followed by the joint correction using the Cholesky factor, which is computed once per step and also
  serves the joint solve.
- All contact arrays have the size `max_contacts`, so the cost of the iteration scales with the buffer size, not
  with the number of active contacts. Choose `max_contacts` close to the expected maximum (about 2 per ball in a
  pile). The `while_loop` is not reverse-mode differentiable; steps of scenes without colliders are
  unchanged.
