# 🚗 The Active Suspension Story
### *A Controls Engineer's Journey — Powered by CrySpace*

---

> *"You're handed a set of physical specs, a noisy sensor, and a deadline.
> Your job: design a controller that keeps a vehicle chassis smooth,
> responsive, and stable — from first equations to production code."*

This is that journey. Every step uses a real CrySpace function. Every result comes
from running [`36_mega_usecase_storyboard.cr`](36_mega_usecase_storyboard.cr).

**→ [▶ Open Interactive Storyboard](https://htmlpreview.github.io/?https://github.com/eltony81/cryspace/blob/main/examples/suspension_storyboard.html)** — rendered in your browser with live charts, equations, and code.

---

## The Problem

A vehicle chassis rides on a spring-mass-damper assembly driven by an electromagnetic actuator.
A position sensor measures chassis displacement — but it lags by 100 ms.
The ride-comfort spec demands: settle within 2 seconds, zero steady-state error.

```
         ╔══════════════════╗
         ║   Road profile   ║  ← disturbance
         ╚══════════════════╝
                  ↕  spring k=2 N/m
                  ↕  damper b=0.5 N·s/m
         ┌────────────────────┐
         │  Chassis  M=1 kg   │ ← state x₁ (position), x₂ (velocity)
         └────────────────────┘
                  ↑ u(t)       ← actuator force [input]
         ┌────────────────────┐
         │   Sensor (100ms)   │ ← y(t) = x₁ + delay [output]
         └────────────────────┘
```

Newton's second law gives the state equations directly:

```
  ẋ₁ =  x₂
  ẋ₂ = −(k/M)·x₁ − (b/M)·x₂ + (1/M)·u
```

---

## Chapter 1 — Write the Physics Down

The first thing any controls engineer does: translate the differential equations into
matrices. CrySpace calls this a `StateSpace` object.

```crystal
a = [[0.0,  1.0],
     [-2.0, -0.5]].to_tensor    # dynamics
b = [[0.0], [1.0]].to_tensor    # input
c = [[1.0,  0.0]].to_tensor    # we can only see position
d = [[0.0]].to_tensor

plant = CrySpace::StateSpace.new(a, b, c, d)
```

One line converts it to a Transfer Function — a classic sanity check:

```crystal
plant_tf = plant.to_transferfunction
# G(s) = 1 / (s² + 0.5s + 2)
```

The sensor delay is modelled as a 1st-order Padé approximation and **chained** into
the plant using the `*` operator — CrySpace handles the series composition automatically:

```crystal
sensor_delay = CrySpace::TransferFunction.pade(0.1, 1)
total_tf     = plant_tf * sensor_delay   # G(s)·H(s)
```

📋 **Result:** 2nd-order plant, open-loop poles at −0.25 ± 1.39j (lightly damped oscillation at ~1.4 rad/s)

---

## Chapter 2 — Is It Even Stable?

Before writing a single line of controller code, we run the **Routh-Hurwitz test** —
a pure algebraic check, no simulation required:

```crystal
rh = total_tf.routh_hurwitz
puts rh[:stable]   # → true ✓
```

Then we check the structural prerequisites for any feedback design to work at all:

```crystal
plant.is_controllable?   # → true  (actuator can reach all states)
plant.is_observable?     # → true  (sensor can reconstruct all states)
```

Now we look at the system's **natural modes** by transforming to Jordan form.
This reveals the bounce frequency as a decoupled oscillator:

```crystal
plant_jcf, T = plant.to_jordan_form
# A_J = [[-0.25, +1.39],
#        [-1.39, -0.25]]  ← complex pair = oscillatory mode
```

The **Hankel Singular Values** [0.843, 0.593] tell us both modes carry significant
energy — neither can be ignored. This will matter in Chapter 7.

---

## Chapter 3 — Sweep the Frequencies

A Bode plot shows exactly where the plant amplifies and where it rolls off.
`bode_data` returns the raw numbers; the HTML storyboard renders them interactively:

```crystal
omega = [0.1, 0.3, 1.0, 3.0, 10.0, 50.0].to_tensor
w, mags_db, phases = plant.bode_data(omega)
```

The stability margins quantify how much room we have before things go unstable:

```crystal
gm, gm_db, pm, w_gc, w_pc = plant.stability_margins
# Phase Margin: 147°  ← enormous, plant is benign open-loop
# Bandwidth:    2.14 rad/s
```

We also compute industry-standard norms that benchmark robustness:

```crystal
plant.h2norm         # → 0.707  (RMS gain under white-noise input)
plant.hinfnorm_exact # → 1.437  (worst-case peak gain, Hamiltonian bisection)
```

---

## Chapter 4 — Design the Optimal Controller

The LQR problem asks: *"What control law minimises the weighted sum of position error
and actuator energy?"* We encode the trade-off in two matrices and let the Riccati
solver find the exact answer:

```crystal
Q = [[100.0, 0.0],   # penalise position error heavily
     [0.0,   1.0]].to_tensor
R = [[0.5]].to_tensor   # allow up to ≈2 N

K, P, cl_poles = plant.lqr(Q, R)
# K ≈ [12.28, 4.68]  → u = −K·x
# Closed-loop poles: −2.59 ± 2.75j  (well-damped, 10× faster than open-loop)
```

For step-reference tracking we need a prefilter so r=1 → y=1 at steady state:

```crystal
Nbar = plant.nbar(K)   # → 14.28
# Control law becomes: u = Nbar·r − K·x̂
```

---

## Chapter 5 — The Observer Problem

The LQR controller needs **both** position and velocity. But the sensor only gives
position. The solution: run a mathematical model of the plant in parallel, feeding it
the same input and correcting its estimate with the measured output.

We design **two** estimators and compare:

### Luenberger Observer (deterministic)
Place observer poles at −15 and −20 — roughly 6× faster than the closed-loop poles,
so estimation error decays before control performance suffers:

```crystal
L = plant.acker_obs([-15.0, -20.0])
# L = [34.5, 280.75]  ← high gain forces fast convergence

obs = CrySpace::LuenbergerObserver.new(plant, L)
obs.update_continuous(y_measured, u_applied, dt)   # RK4 integration step
```

### Kalman Filter (optimal under noise)
The Kalman gain balances process noise against sensor noise — tunable via covariance matrices:

```crystal
Q_noise = [[0.5, 0.0], [0.0, 0.1]].to_tensor   # model uncertainty
R_noise = [[0.05]].to_tensor                    # sensor noise variance

L_lqe = plant.lqe(Q_noise, R_noise)   # optimal Kalman gain

kf = CrySpace::KalmanFilter.new(plant_d, Q_noise, R_noise)
kf.predict(u)   # propagate state estimate
kf.update(y)    # correct with new measurement
```

📋 **Observer convergence:** initial 0.5m estimation error → < 0.001m within 0.5s

---

## Chapter 6 — Going Digital

The real controller runs on a microcontroller at **20 Hz**. We discretize the
continuous-time plant using four different methods and compare:

```crystal
dt = 0.05   # 20 Hz

plant.sample(dt)                                  # ZOH    (exact for step inputs)
plant.sample(dt, method: :tustin)                 # Bilinear (stable → stable)
plant.sample(dt, method: :prewarped, omega_c: 5.0) # Tustin with frequency warp
plant.sample(dt, method: :matched)                # Pole-zero matched
```

| Method | A[0,0] | Best for |
|--------|--------|----------|
| ZOH | 0.997522 | General purpose, exact hold |
| Tustin | 0.997534 | Digital filter design |
| Pre-warped | 0.997508 | Matching gain at specific ω |
| Matched | 1.970374 | Preserving pole-zero locations |

The discrete LQR can then be run directly on the digital plant:

```crystal
Kd, _, _ = plant_d.dlqr(Q, R)   # discrete-time optimal gain
```

---

## Chapter 7 — The Plant Has Too Many States

The combined plant + sensor delay system is **3rd order**. For the embedded target, a
2nd-order approximation is sufficient. We use **Balanced Truncation**:

```crystal
sys3    = total_tf.to_statespace   # 3 states
sys_bt  = sys3.balred(2)           # keep 2 most controllable/observable modes
```

The balanced realisation reveals *why* we can discard a mode — the third Hankel
singular value is tiny (0.002 vs 0.859 and 0.612):

```crystal
sys_bal, T, Ti = sys3.balreal
Wc = sys_bal.gram(:c)   # controllability Gramian
Wo = sys_bal.gram(:o)   # observability Gramian
# In balanced form: Wc = Wo = diag(σ₁, σ₂, σ₃)
# → σ₃ ≈ 0.002, safely truncated
```

DC gain error after reduction: **< 1%** — the reduced model faithfully represents
the plant's static behaviour.

---

## Chapter 8 — A Simpler Fallback Controller

Not every deployment warrants a full LQR. As a fallback, we apply classical
PID tuning using two well-known empirical rules:

```crystal
# Ziegler-Nichols — from measured ultimate gain/period
zn = CrySpace::Tuning.pid_tune_zn(ku: 8.0, tu: 1.2, type: :pid)
# → Kp=4.8, Ki=8.0, Kd=0.72

# Cohen-Coon — from first-order-plus-dead-time fit
cc = CrySpace::Tuning.pid_tune_cc(gp: 0.5, tau: 1.5, theta: 0.1, type: :pid)
```

The actuator saturates at ±3 N. Without anti-windup, the integrator winds up
and causes massive overshoot on release. The `PIDController` clamps output
**and** prevents integrator accumulation past the saturation boundary:

```crystal
pid = CrySpace::PIDController.new(
  kp: 4.8, ki: 8.0, kd: 0.72,
  tf: 0.02,        # derivative low-pass filter time constant
  u_min: -3.0,     # saturation with clamping anti-windup built in
  u_max:  3.0
)

u = pid.update(error, dt)   # returns clamped output
```

---

## Chapter 9 — Nonlinear Reality Check

The actuator has **two** nonlinearities: a saturation limit and a dead-zone below 0.3 N
where the electromagnetic force is too weak to overcome stiction.

**Describing Function Analysis** models these as amplitude-dependent gains —
allowing classical frequency-domain tools to predict limit cycles:

```crystal
# Saturation: gain drops from 1.0 once amplitude exceeds limit
CrySpace::Nonlinear.describing_function_saturation(4.0, 3.0)  # → 0.856
CrySpace::Nonlinear.describing_function_saturation(6.0, 3.0)  # → 0.609

# Dead-zone: gain rises from 0 as amplitude exceeds threshold
CrySpace::Nonlinear.describing_function_deadzone(0.1, 0.3)   # → 0.0   (below threshold)
CrySpace::Nonlinear.describing_function_deadzone(1.0, 0.3)   # → 0.624
```

📋 **Insight:** At large input amplitudes (>2× the saturation limit), the effective
gain falls below 0.7 — reduce proportional gains accordingly in the nonlinear regime.

---

## Chapter 10 — What If We Didn't Know the Model?

Before deploying to a new vehicle platform, the engineer must **identify** the plant
from measurement data. CrySpace provides the full identification pipeline:

```crystal
# 1. Design a persistent excitation signal (maximal-length binary sequence)
prbs = CrySpace::Ident.prbs(4)   # 15-sample PRBS

# 2. Swept-frequency chirp for broadband frequency response
chirp = CrySpace::Ident.chirp(t, f_start: 0.0, f_end: 5.0, amplitude: 1.0)

# 3. Collect noisy step-response I/O data from the rig
# ... (apply step, sample at dt=0.1s) ...

# 4. ARX least-squares: estimate TF parameters from I/O data
sys_id = CrySpace::Ident.least_squares_estimation(u_data, y_data, 1, 0.1)
# Returns a discrete TransferFunction with estimated coefficients

# 5. Realization from impulse response (Ho-Kalman algorithm)
sys_hk = CrySpace::Ident.ho_kalman(impulse_response, 1, 1, 1, 0.1)
# A[0,0] ≈ 0.85 matches the known plant exactly ✓
```

---

## Chapter 11 — The Road Has Rubber Bump-Stops

The real suspension has **cubic spring stiffening** from rubber bump-stops —
a hard nonlinearity that linear models cannot capture. We simulate it with a
custom ODE and let CrySpace's adaptive solver handle the step-size automatically:

```crystal
# Nonlinear EOM: stiffness grows with displacement squared
f_nl = ->(x : Float64Tensor, t : Float64) {
  x1, x2 = x[0].value, x[1].value
  [x2, -(2.0 + 0.3*x1*x1)*x1 - 0.5*x2].to_tensor
}

# Fixed-step RK4 (fast, predictable)
_, states_rk4  = CrySpace::Solver.rk4(f_nl, x0, {0.0, 1.0}, 0.05)

# Adaptive RK45 — automatically tightens steps near the transient
_, states_rk45 = CrySpace::Solver.rk45(f_nl, x0, {0.0, 1.0}, tol: 1e-7)

# Both agree: x₁(1.0s) = 0.12945 m  ✓
```

For **adaptive control** — when the spring constant might drift over time — we
apply **Model Reference Adaptive Control (MRAC)**:

```crystal
xs, xm, us = CrySpace::AdaptiveNonlinear.mrac_simulate(
  -1.0, 2.0, -3.0, 3.0, 1.0, 1.0, r_sig, t_vec
)
```

---

## Chapter 12 — The Full Picture

Everything comes together: LQR controller, Luenberger observer, nonzero initial
conditions. The `simulate_observer` call handles the coupled plant+observer
differential equations for us:

```crystal
t    = Float64Tensor.linear_space(0.0, 2.0, 80)
x0   = [[0.5], [0.0]].to_tensor    # chassis displaced 0.5m
x0_e = [[0.0], [0.0]].to_tensor    # observer has no prior knowledge

t_out, x, x_hat, y, u = plant.simulate_observer(
  t, K, L_luenberger, x0: x0, x0_est: x0_e
)
```

```
  time │ x₁ (true) │ x̂₁ (est) │ error
──────────────────────────────────────────
   0.0 │  0.50000  │  0.00000  │ 0.50000  ← observer starts blind
   0.5 │ -0.12514  │ -0.12504  │ 0.00010  ← observer has converged ✓
   1.0 │ -0.07428  │ -0.07428  │ 0.00000
   2.0 │  0.00624  │  0.00624  │ 0.00000  ← plant at rest ✓
```

And `stepinfo` gives the official closed-loop performance metrics:

```crystal
info = plant.feedback([[1.0]].to_tensor).stepinfo(n_steps: 200)
# Rise time:     0.70 s
# Settling time: 16.6 s   (unity feedback — uncompensated)
# Overshoot:     63.8 %   → LQR reduces this dramatically
```

---

## Chapter 13 — Ship It

The final deliverable is a browser-ready dashboard — zero build steps:

```crystal
plant.step_plot("suspension_step.html")   # interactive step response
plant.bode_plot("suspension_bode.html")   # interactive Bode diagram
```

Open any of these in a browser. Share with the team. No MATLAB licence required.

---

## Running the Example

```bash
cd /path/to/cryspace

# Run the full storyboard (all 13 chapters, ~3 seconds)
crystal run examples/36_mega_usecase_storyboard.cr

# Open the interactive HTML narrative
xdg-open examples/suspension_storyboard.html

# Open the generated dashboards
xdg-open suspension_step_response.html
xdg-open suspension_bode_diagram.html
```

---

## Functions Demonstrated

| Area | CrySpace Functions |
|------|-------------------|
| **Modelling** | `StateSpace.new` · `to_transferfunction` · `TransferFunction.pade` · `dcgain` · `poles` |
| **Stability** | `routh_hurwitz` · `is_controllable?` · `is_observable?` · `to_jordan_form` · `to_control_canonical_form` · `hsvd` |
| **Frequency** | `bode_data` · `stability_margins` · `bandwidth` · `h2norm` · `hinfnorm_exact` · `peak_gain` · `nyquist_data` |
| **Control** | `lqr` · `dlqr` · `nbar` · `h2syn` · `hinfsyn` · `feedback` |
| **Estimation** | `lqe` · `dlqe` · `acker_obs` · `LuenbergerObserver` · `KalmanFilter` · `UnscentedKalmanFilter` |
| **Discretization** | `sample` (ZOH / Tustin / pre-warped / matched) · `to_continuous` |
| **Reduction** | `balred` · `hankel_reduction` · `balreal` · `gram` · `minreal` · `fw_balred` |
| **PID** | `pid_tune_zn` · `pid_tune_cc` · `PIDController` |
| **Nonlinear** | `describing_function_saturation` · `describing_function_deadzone` |
| **Identification** | `prbs` · `chirp` · `least_squares_estimation` · `ho_kalman` · `era` · `levy_fit` |
| **Simulation** | `Solver.rk4` · `Solver.rk45` · `lsim` · `simulate_observer` · `mrac_simulate` |
| **Metrics** | `stepinfo` · `step_response` |
| **Plotting** | `step_plot` · `bode_plot` |

---

## What's Next?

Explore more CrySpace examples:

| File | Topic |
|------|-------|
| `01_basic_statespace.cr` | State-space fundamentals |
| `05_lqr_control.cr` | LQR deep-dive |
| `12_kalman_filter.cr` | Kalman filtering |
| `20_system_identification.cr` | Full identification workflow |
| `36_mega_usecase_storyboard.cr` | **← You are here** |

---

*Built with [CrySpace](https://github.com/eltony81/cryspace) — Control Systems in Crystal*
