require "../src/cryspace"

puts "================================================================================"
puts "  CrySpace Complete Storyboard Tutorial"
puts "  Case Study: Active Vehicle Suspension Control System Design"
puts "================================================================================"
puts <<-STORY
STORY:
  A controls engineer is tasked with designing a feedback control system for
  an active vehicle suspension. The physical rig has a spring-mass-damper plant
  driven by an electromagnetic actuator, a noisy position sensor, and strict
  ride-comfort requirements. This tutorial follows every step of the process:
  system modeling → analysis → control design → observer → simulation → plotting.

  Physical Parameters:
    Mass    M = 1 kg
    Damper  b = 0.5 N·s/m
    Spring  k = 2 N/m
    Input   u = actuator force [N]
    Output  y = chassis position [m]
STORY


# ==============================================================================
# Chapter 1 — Plant Modeling & Representation
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 1 · Plant Modeling & Representation"
puts "═══════════════════════════════════════════════════════════════"

# State-Space form derived from Newton's 2nd Law:
#   ẋ₁ = x₂                      (velocity)
#   ẋ₂ = -k/M·x₁ - b/M·x₂ + u/M (acceleration)
a_plant = [[0.0,  1.0],
           [-2.0, -0.5]].to_tensor
b_plant = [[0.0], [1.0]].to_tensor
c_plant = [[1.0,  0.0]].to_tensor
d_plant = [[0.0]].to_tensor

plant = CrySpace::StateSpace.new(a_plant, b_plant, c_plant, d_plant)
puts "▶ SS plant created:  #{plant.n_states} states, #{plant.n_inputs} input, #{plant.n_outputs} output"
puts "  Eigenvalues (open-loop poles): #{plant.poles.map { |p| "(#{p.real.round(4)}, #{p.imag.round(4)}j)" }.join(", ")}"

# Convert to Transfer Function G(s) = Y(s)/U(s)
plant_tf = plant.to_transferfunction
puts "\n▶ Transfer Function G(s):"
puts plant_tf

# DC gain: static force-to-displacement ratio (spring stiffness compliance)
dc = plant.dcgain
puts "▶ DC gain (static compliance x/F at s=0): #{dc[0,0].value.round(4)} m/N"

# Sensor lag: The position sensor introduces a 100ms measurement delay
puts "\n▶ Modelling 0.1s sensor delay via 1st-order Pade approximation:"
sensor_delay_tf = CrySpace::TransferFunction.pade(0.1, 1)
puts sensor_delay_tf

# Total measured system: G(s)·H_delay(s)
sys_with_delay_tf = plant_tf * sensor_delay_tf
puts "▶ Composite G_measured(s) = G(s) · H_delay(s):"
puts sys_with_delay_tf

# Check poles/zeros of composite system
puts "▶ Composite transmission zeros: #{plant_tf.zeros.map { |z| z.real.round(3) }.join(", ")}"


# ==============================================================================
# Chapter 2 — Stability & Canonical Forms
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 2 · Stability Analysis & Canonical Realizations"
puts "═══════════════════════════════════════════════════════════════"

# Routh-Hurwitz stability test on composite open-loop system
rh = sys_with_delay_tf.routh_hurwitz
puts "▶ Routh-Hurwitz (open-loop composite system): stable = #{rh[:stable]}"

# Check if the plain plant is controllable and observable
puts "\n▶ Plant controllability: #{plant.is_controllable?}"
puts "▶ Plant observability:   #{plant.is_observable?}"

# Transform to Control Canonical Form (reveals characteristic polynomial)
plant_ccf, t_ccf = plant.to_control_canonical_form
puts "\n▶ Control Canonical Form A matrix:"
puts plant_ccf.a

# Transform to Jordan / Modal Canonical Form (decoupled modal dynamics)
plant_jcf, t_jcf = plant.to_jordan_form
puts "\n▶ Jordan Canonical Form A matrix (modal dynamics):"
puts plant_jcf.a

# Hankel Singular Values tell us how 'important' each mode is
hsv = plant.hsvd
puts "\n▶ Hankel Singular Values: #{hsv.map { |h| h.round(5) }.join(", ")}"


# ==============================================================================
# Chapter 3 — Frequency Domain Analysis
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 3 · Frequency Domain Analysis"
puts "═══════════════════════════════════════════════════════════════"

# Bode data over relevant frequency range 0.1 – 100 rad/s
omega_vals = [0.1, 0.3, 1.0, 3.0, 10.0, 50.0]
omega_log = omega_vals.to_tensor
w_out, mags_db, phases_deg = plant.bode_data(omega_log)

puts "▶ Bode frequency response:"
omega_log.size.times do |i|
  puts "  ω = #{w_out[i].value.round(2).to_s.rjust(6)} rad/s │ " \
       "#{mags_db[i].round(2).to_s.rjust(8)} dB │ #{phases_deg[i].round(1).to_s.rjust(8)} °"
end

# Stability margins
gm, gm_db, pm, w_gc, w_pc = plant.stability_margins
puts "\n▶ Open-Loop Stability Margins:"
puts "  Gain Margin:  #{gm_db.infinite? ? "∞" : gm_db.round(2)} dB   (at ω_pc = #{w_pc < 0 ? "N/A" : w_pc.round(3)} rad/s)"
puts "  Phase Margin: #{pm.round(2)} °  (at ω_gc = #{w_gc.round(3)} rad/s)"
puts "  Bandwidth:    #{plant.bandwidth.round(3)} rad/s"

# Peak gain (H∞ norm of open-loop)
puts "\n▶ H∞ norm (peak gain): #{plant.peak_gain.round(4)}"
puts "▶ H2 norm:             #{plant.h2norm.round(4)}"

# Exact H∞ norm via Hamiltonian bisection (MIMO-grade)
puts "▶ Exact H∞ norm (Hamiltonian method): #{plant.hinfnorm_exact.round(4)}"


# ==============================================================================
# Chapter 4 — LQR State-Feedback Design & Tracking
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 4 · LQR State-Feedback & Prefilter Design"
puts "═══════════════════════════════════════════════════════════════"

# LQR: trade off position tracking accuracy vs control effort
#   Q penalises position error, R limits actuator power
q_lqr = [[100.0, 0.0], [0.0, 1.0]].to_tensor
r_lqr = [[0.5]].to_tensor
k_lqr, p_care, cl_poles_lqr = plant.lqr(q_lqr, r_lqr)
puts "▶ LQR gain K:              #{k_lqr.to_a.flatten.map { |x| x.round(4) }.join(", ")}"
puts "▶ Closed-loop poles (LQR): #{cl_poles_lqr.map { |p| "(#{p.real.round(3)},#{p.imag.round(3)}j)" }.join(", ")}"

# Reference prefilter Nbar so r=1 → y→1 at steady-state
n_bar = plant.nbar(k_lqr)
puts "▶ Prefilter Nbar:          #{n_bar[0,0].value.round(4)}"

# Sensitivity and complementary sensitivity functions
# Build unity-gain static feedback controller for comparison
cl_plant = plant.feedback([[1.0]].to_tensor)
puts "\n▶ Unity-feedback closed-loop order: #{cl_plant.n_states} states"
puts "▶ Closed-loop DC gain:             #{cl_plant.dcgain[0,0].value.round(4)}"

# H2 synthesis (alternative optimal state feedback)
c_z  = [[1.0, 0.0]].to_tensor   # position performance output
d_zu = [[0.5]].to_tensor         # control penalty
k_h2, _ = plant.h2syn(c_z, d_zu)
puts "\n▶ H2-optimal state-feedback gain: #{k_h2.to_a.flatten.map { |x| x.round(4) }.join(", ")}"


# ==============================================================================
# Chapter 5 — Observer Design: Kalman & Luenberger
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 5 · State Estimation: LQE / Luenberger Observer"
puts "═══════════════════════════════════════════════════════════════"

# LQE (Kalman Estimator) – optimal observer gain given process and sensor noise
q_noise = [[0.5, 0.0], [0.0, 0.1]].to_tensor  # process noise covariance
r_noise = [[0.05]].to_tensor                   # measurement noise variance
l_lqe   = plant.lqe(q_noise, r_noise)
puts "▶ LQE Kalman Observer gain L: [#{l_lqe[0,0].value.round(4)}, #{l_lqe[1,0].value.round(4)}]"

# Observer poles from LQE should be well inside LHP
obs_eigvals = (a_plant - l_lqe.matmul(c_plant)).eigvals_c.to_a
puts "▶ Observer closed-loop eigenvalues: #{obs_eigvals.map { |e| "(#{e.real.round(3)},#{e.imag.round(3)}j)" }.join(", ")}"

# Luenberger Observer (deterministic pole-placement alternative)
l_lbg = plant.acker_obs([-15.0, -20.0])
puts "\n▶ Luenberger gain (poles @ -15, -20): [#{l_lbg[0,0].value.round(4)}, #{l_lbg[1,0].value.round(4)}]"

# Create and step Luenberger observer object (continuous RK4 update)
luenberger = CrySpace::LuenbergerObserver.new(plant, l_lbg, x0_hat: [[0.0], [0.0]].to_tensor)
luenberger.update_continuous([[0.4]].to_tensor, [[0.0]].to_tensor, 0.05)
puts "▶ Luenberger x_hat after 1 RK4 step (y=0.4, u=0): [#{luenberger.x_hat[0,0].value.round(5)}, #{luenberger.x_hat[1,0].value.round(5)}]"

# Discrete Kalman Filter on the ZOH-discretised plant
dt = 0.05
plant_d = plant.sample(dt)
kf = CrySpace::KalmanFilter.new(plant_d, q_noise, r_noise)
kf.predict([[0.0]].to_tensor)
kf.update([[0.5]].to_tensor)
puts "\n▶ Discrete Kalman Filter estimate after predict+update (y=0.5): [#{kf.x[0,0].value.round(5)}, #{kf.x[1,0].value.round(5)}]"


# ==============================================================================
# Chapter 6 — Discretization Methods Comparison
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 6 · Discretization Methods"
puts "═══════════════════════════════════════════════════════════════"

plant_zoh     = plant.sample(dt)
plant_tustin  = plant.sample(dt, method: :tustin)
plant_prewarp = plant.sample(dt, method: :prewarped, omega_c: 5.0)
plant_matched = plant.sample(dt, method: :matched)

puts "▶ ZOH A₁₁:          #{plant_zoh.a[0,0].value.round(6)}"
puts "▶ Tustin A₁₁:       #{plant_tustin.a[0,0].value.round(6)}"
puts "▶ Pre-warped A₁₁:   #{plant_prewarp.a[0,0].value.round(6)}"
puts "▶ Matched A₁₁:      #{plant_matched.a[0,0].value.round(6)}"

# Verify round-trip: continuous → discrete → continuous recovers A
plant_back = plant_zoh.to_continuous
puts "\n▶ D2C round-trip (ZOH → continuous) A₁₁: #{plant_back.a[0,0].value.round(4)}"
puts "  (Original A₁₁ = #{a_plant[0,0].value.round(4)})"


# ==============================================================================
# Chapter 7 — Model Order Reduction
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 7 · Model Order Reduction"
puts "═══════════════════════════════════════════════════════════════"

# Build a higher-order system: cascade plant with sensor lag
sys_high = sys_with_delay_tf.to_statespace
puts "▶ High-order plant+delay system: #{sys_high.n_states} states"

# Balanced Truncation: keep 2 most controllable/observable modes
sys_bt = sys_high.balred(2)
puts "▶ After Balanced Truncation → #{sys_bt.n_states} states"
puts "  DC gain original: #{sys_high.dcgain[0,0].value.round(4)}"
puts "  DC gain reduced:  #{sys_bt.dcgain[0,0].value.round(4)}"

# Hankel Norm Approximation (same API, slightly different approximation)
sys_hn = sys_high.hankel_reduction(2)
puts "▶ After Hankel Norm Approx → #{sys_hn.n_states} states"

# Balanced realisation
sys_bal, t_bal, t_bal_inv = sys_high.balreal
puts "\n▶ Balanced realisation computed. Controllability Gramian diagonal check:"
wc = sys_bal.gram(:c)
wo = sys_bal.gram(:o)
sys_high.n_states.times { |i| puts "  σ_#{i+1}: Wc[#{i},#{i}] = #{wc[i,i].value.round(5)}  Wo[#{i},#{i}] = #{wo[i,i].value.round(5)}" }

# Minimal realisation: cancel any hidden pole-zero pairs
sys_min = sys_high.minreal
puts "\n▶ Minimal realisation order: #{sys_min.n_states} (from #{sys_high.n_states})"


# ==============================================================================
# Chapter 8 — PID Design & Anti-Windup
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 8 · PID Tuning & Anti-Windup"
puts "═══════════════════════════════════════════════════════════════"

# Ziegler-Nichols ultimate-oscillation method
zn = CrySpace::Tuning.pid_tune_zn(ku: 8.0, tu: 1.2, type: :pid)
puts "▶ ZN PID: Kp=#{zn[:kp].round(3)}, Ki=#{zn[:ki].round(3)}, Kd=#{zn[:kd].round(3)}"

# Cohen-Coon FOPDT method (first-order-plus-dead-time process model)
cc = CrySpace::Tuning.pid_tune_cc(gp: 0.5, tau: 1.5, theta: 0.1, type: :pid)
puts "▶ CC PID: Kp=#{cc[:kp].round(3)}, Ki=#{cc[:ki].round(3)}, Kd=#{cc[:kd].round(3)}"

# PID Controller with anti-windup clamping
pid = CrySpace::PIDController.new(
  kp: zn[:kp], ki: zn[:ki], kd: zn[:kd],
  tf: 0.02, u_min: -3.0, u_max: 3.0
)
# Compute a few PID steps simulating an error signal
errors   = [0.5, 0.38, 0.25, 0.12, 0.04]
controls = errors.map { |e| pid.update(e, dt).round(4) }
puts "\n▶ PID control outputs for decreasing error trajectory:"
errors.each_with_index { |e, i| puts "  e=#{e} → u=#{controls[i]}" }


# ==============================================================================
# Chapter 9 — Describing Functions & Nonlinear Analysis
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 9 · Nonlinear Analysis via Describing Functions"
puts "═══════════════════════════════════════════════════════════════"

# Saturation: actuator limited to ±3 N
[1.0, 2.5, 4.0, 6.0].each do |amp|
  df = CrySpace::Nonlinear.describing_function_saturation(amp, 3.0)
  puts "  Amplitude #{amp.to_s.rjust(4)} N → Saturation DF gain = #{df.round(5)}"
end

# Dead-zone: threshold before actuator engages
puts "\n▶ Dead-zone (half-width 0.3 N) describing function:"
[0.1, 0.5, 1.0, 2.0].each do |amp|
  df = CrySpace::Nonlinear.describing_function_deadzone(amp, 0.3)
  puts "  Amplitude #{amp.to_s.rjust(4)} N → Deadzone DF gain = #{df.round(5)}"
end


# ==============================================================================
# Chapter 10 — System Identification & Excitation Signals
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 10 · System Identification"
puts "═══════════════════════════════════════════════════════════════"

# Generate PRBS excitation signal for plant identification
prbs = CrySpace::Ident.prbs(4)   # 4-stage PRBS: length = 2^4 - 1 = 15
puts "▶ PRBS excitation (4 stage, #{prbs.size} samples): #{prbs.to_a.map { |v| v.round.to_i }.join(" ")}"

# Chirp sweep signal (0→5 Hz over 1 second)
t_ident = Float64Tensor.linear_space(0.0, 1.0, 16)
chirp = CrySpace::Ident.chirp(t_ident, 0.0, 5.0, 10.0)
puts "▶ Chirp signal first 6 samples: #{chirp.to_a.first(6).map { |v| v.round(3) }.join(", ")}"

# Simulate plant with step input to collect I/O data for ARX identification
plant_d_id = plant.sample(0.1)
r = Random.new(12345)
u_id = Float64Tensor.zeros([40])
y_id = Float64Tensor.zeros([40])
u_id[0] = 0.0
y_id[0] = 0.0
(1...40).each do |k|
  u_id[k] = k > 5 ? 1.0 : 0.0   # step at k=5
  y_id[k] = plant_d_id.a[0,0].value * y_id[k-1].value +
             plant_d_id.b[0,0].value * u_id[k-1].value +
             0.01 * r.rand(-1.0..1.0)
end
sys_ident = CrySpace::Ident.least_squares_estimation(u_id, y_id, 1, 0.1)
puts "\n▶ ARX system identification from noisy step data:"
puts "  Identified TF:"
puts sys_ident

# Impulse response from a known discrete system → Ho-Kalman realization
ir = Float64Tensor.zeros([8])
(0...8).each { |k| ir[k] = k == 0 ? 0.0 : 0.85 ** (k-1) }
sys_hk = CrySpace::Ident.ho_kalman(ir, 1, 1, 1, 0.1)
puts "\n▶ Ho-Kalman Realization from impulse response:"
puts "  A = #{sys_hk.a[0,0].value.round(5)}  (expected ≈ 0.85)"


# ==============================================================================
# Chapter 11 — Nonlinear & Adaptive Simulation
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 11 · Nonlinear ODE Solvers & Adaptive Control"
puts "═══════════════════════════════════════════════════════════════"

# RK4 simulation of the nonlinear spring-damper with cubic stiffness
# ẋ₁ = x₂
# ẋ₂ = -(2 + 0.3·x₁²)·x₁ - 0.5·x₂ + u
f_nl = ->(x : Float64Tensor, t : Float64) {
  x1 = x[0].value
  x2 = x[1].value
  dx1 = x2
  dx2 = -(2.0 + 0.3*x1*x1)*x1 - 0.5*x2
  [dx1, dx2].to_tensor
}
x0_nl = [0.5, 0.0].to_tensor
_, states_rk4 = CrySpace::Solver.rk4(f_nl, x0_nl, {0.0, 1.0}, 0.05)
puts "▶ RK4 nonlinear sim: x₁(1.0s) = #{states_rk4.last[0].value.round(5)} m"

# Adaptive RK45 (error-controlled step-size) on the same system
_, states_rk45 = CrySpace::Solver.rk45(f_nl, x0_nl, {0.0, 1.0}, tol: 1e-7)
puts "▶ RK45 adaptive sim: x₁(1.0s) = #{states_rk45.last[0].value.round(5)} m"
puts "  (Both methods should agree to high precision)"

# MRAC: adapt a 1D controller to track a reference model
r_sig = ->(t : Float64) { Math.sin(2.0 * Math::PI * t) }
t_mrac = Float64Tensor.linear_space(0.0, 1.0, 21)
xs_m, xms_m, us_m = CrySpace::AdaptiveNonlinear.mrac_simulate(-1.0, 2.0, -3.0, 3.0, 1.0, 1.0, r_sig, t_mrac)
puts "\n▶ MRAC simulation over 1s: final plant state  = #{xs_m[-1].value.round(5)}"
puts "                            final model state = #{xms_m[-1].value.round(5)}"


# ==============================================================================
# Chapter 12 — Closed-Loop Observer Simulation (LQG-like)
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 12 · Full Closed-Loop Observer Simulation"
puts "═══════════════════════════════════════════════════════════════"

t_sim = Float64Tensor.linear_space(0.0, 2.0, 80)
x0_sim   = [[0.5], [0.0]].to_tensor    # initial displacement 0.5 m
x0_est   = [[0.0], [0.0]].to_tensor    # observer starts at origin

t_out, x_out, x_est_out, y_out, u_out = plant.simulate_observer(
  t_sim, k_lqr, l_lbg, x0: x0_sim, x0_est: x0_est
)

step_times = [0, 20, 40, 60, 79]
puts "▶ Closed-loop response under LQR + Luenberger Observer:"
puts "  #{"time".rjust(6)} │ #{"x (true)".rjust(10)} │ #{"x̂ (est)".rjust(10)} │ #{"error".rjust(10)}"
step_times.each do |i|
  xt = x_out[i, 0].value
  xe = x_est_out[i, 0].value
  puts "  #{t_sim[i].value.round(3).to_s.rjust(6)} │ #{xt.round(5).to_s.rjust(10)} │ #{xe.round(5).to_s.rjust(10)} │ #{(xt - xe).abs.round(6).to_s.rjust(10)}"
end

# Step response metrics using stepinfo
info = plant.feedback([[1.0]].to_tensor).stepinfo(n_steps: 200)
puts "\n▶ Closed-loop step response metrics (unity feedback):"
puts "  Rise time:         #{info.rise_time.round(4)} s"
puts "  Settling time:     #{info.settling_time.round(4)} s"
puts "  Overshoot:         #{info.overshoot.round(3)} %"
puts "  Steady-state:      #{info.steady_state_value.round(5)}"


# ==============================================================================
# Chapter 13 — Dashboard Plotting
# ==============================================================================
puts "\n═══════════════════════════════════════════════════════════════"
puts " Chapter 13 · Interactive HTML Dashboard Generation"
puts "═══════════════════════════════════════════════════════════════"

step_html = "suspension_step_response.html"
bode_html = "suspension_bode_diagram.html"

plant.step_plot(step_html)
plant.bode_plot(bode_html)
puts "▶ Step Response Dashboard  → #{step_html}"
puts "▶ Bode Diagram Dashboard   → #{bode_html}"
puts "  Open these files in any browser to see interactive Chart.js plots."

puts "\n================================================================================"
puts "  Tutorial Complete — 13 Chapters, 30+ CrySpace functions demonstrated!"
puts "================================================================================"
