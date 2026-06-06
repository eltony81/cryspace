require "../src/cryspace"

puts "=================================================="
puts "CrySpace: Showcasing 19 Advanced Control Suite"
puts "=================================================="

# 1. Routh-Hurwitz Stability
puts "\n1. Routh-Hurwitz Stability Criterion:"
tf_stable = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 3.0, 2.0].to_tensor)
res_rh = tf_stable.routh_hurwitz
puts "Denominator s^2 + 3s + 2 is stable: #{res_rh[:stable]}"

# 2. Jury Stability Test
puts "\n2. Jury Stability Criterion:"
tf_discrete = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, -0.5, 0.06].to_tensor, 0.1)
res_jury = tf_discrete.jury_test
puts "Discrete denominator z^2 - 0.5z + 0.06 is stable: #{res_jury[:stable]}"

# 3. Relative Gain Array (RGA)
puts "\n3. Relative Gain Array (RGA):"
sys_mimo = CrySpace::StateSpace.new(
  [[-1.0, 0.0], [0.0, -2.0]].to_tensor,
  [[1.0, 0.0], [0.0, 1.0]].to_tensor,
  [[2.0, 1.0], [1.0, 2.0]].to_tensor,
  [[0.0, 0.0], [0.0, 0.0]].to_tensor
)
rga_dc = sys_mimo.rga(0.0)
puts "MIMO DC RGA Matrix:"
puts rga_dc

# 4. Bilinear Discretization with Frequency Pre-Warping
puts "\n4. Bilinear Discretization with Pre-Warping:"
sys_c = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
sys_d = sys_c.sample(dt: 0.1, method: :prewarped, omega_c: 5.0)
puts "Pre-warped Discrete System A matrix:"
puts sys_d.a

# 5. Ho-Kalman Realization
puts "\n5. Ho-Kalman Realization:"
ir = [0.0, 1.0, 0.5, 0.25, 0.125, 0.0625].to_tensor
sys_ho = CrySpace::Ident.ho_kalman(ir, 1, 1, 1, 0.1)
puts "Ho-Kalman Realization A matrix (expected ~0.5):"
puts sys_ho.a

# 6. Kung's Realization
puts "\n6. Kung's Realization:"
sys_kung = CrySpace::Ident.kung(ir, 1, 1, 1, 0.1)
puts "Kung's Realization A matrix (expected ~0.5):"
puts sys_kung.a

# 7. Levy's Frequency-Domain Fitting
puts "\n7. Levy's Frequency-Domain Fitting:"
omega = [0.1, 1.0, 10.0].to_tensor
# Target G(s) = 2 / (s + 3)
resp = Tensor(Complex, CPU(Complex)).zeros([3])
3.times do |i|
  w = omega[i].value
  den = w * w + 9.0
  resp[i] = Complex.new(6.0 / den, -2.0 * w / den)
end
tf_fit = CrySpace::Ident.levy_fit(omega, resp, 0, 1)
puts "Fitted TransferFunction:"
puts tf_fit

# 8. Extended Kalman Filter (EKF)
puts "\n8. Extended Kalman Filter (EKF):"
f = ->(x : Float64Tensor, u : Float64Tensor?) { (x * x) + (u ? u : 0.0) }
h = ->(x : Float64Tensor) { x * x * x }
jf = ->(x : Float64Tensor, u : Float64Tensor?) { x * 2.0 }
jh = ->(x : Float64Tensor) { (x * x) * 3.0 }
ekf = CrySpace::ExtendedKalmanFilter.new(f, h, jf, jh, [[0.01]].to_tensor, [[0.01]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor)
ekf.predict([[0.5]].to_tensor)
ekf.update([[3.375]].to_tensor)
puts "EKF updated state: #{ekf.x[0,0].value}"

# 9. Unscented Kalman Filter (UKF)
puts "\n9. Unscented Kalman Filter (UKF):"
ukf = CrySpace::UnscentedKalmanFilter.new(f, h, [[0.01]].to_tensor, [[0.01]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor)
ukf.predict([[0.5]].to_tensor)
ukf.update([[3.375]].to_tensor)
puts "UKF updated state: #{ukf.x[0,0].value}"

# 10. Notch Filter Design
puts "\n10. Notch Filter Design:"
notch = CrySpace::TransferFunction.notch(10.0, 0.1)
puts "Notch filter tf (rejects 10 rad/s):"
puts notch

# 11. PRBS Generator
puts "\n11. PRBS Generator:"
prbs = CrySpace::Ident.prbs(4)
puts "PRBS signal length #{prbs.size}:"
puts prbs

# 12. Chirp Generator
puts "\n12. Chirp Generator:"
t = Float64Tensor.linear_space(0.0, 1.0, 10)
chirp = CrySpace::Ident.chirp(t, 0.0, 1.0, 10.0)
puts "Swept-frequency chirp signal:"
puts chirp

# 13. Ziegler-Nichols Closed-Loop PID Tuning
puts "\n13. Ziegler-Nichols Closed-Loop Tuning:"
zn_gains = CrySpace::Tuning.pid_tune_zn(ku: 2.5, tu: 3.0, type: :pid)
puts "Gains: Kp = #{zn_gains[:kp]}, Ki = #{zn_gains[:ki]}, Kd = #{zn_gains[:kd]}"

# 14. Cohen-Coon PID Tuning
puts "\n14. Cohen-Coon PID Tuning:"
cc_gains = CrySpace::Tuning.pid_tune_cc(gp: 2.0, tau: 10.0, theta: 1.5, type: :pid)
puts "Gains: Kp = #{cc_gains[:kp]}, Ki = #{cc_gains[:ki]}, Kd = #{cc_gains[:kd]}"

# 15. Exact H-infinity Norm Solver
puts "\n15. Exact H-infinity Norm Solver:"
hinf = sys_mimo.hinfnorm_exact
puts "Exact H-infinity norm (expected ~2.422): #{hinf}"

# 16. Singular Perturbation Model Reduction (Residualization)
puts "\n16. Singular Perturbation Model Reduction:"
sys_large = CrySpace::StateSpace.new(
  [[-1.0, 0.0], [0.0, -100.0]].to_tensor,
  [[1.0], [10.0]].to_tensor,
  [[1.0, 1.0]].to_tensor,
  [[0.0]].to_tensor
)
sys_r = sys_large.residual_reduction(1)
puts "Reduced system DC gain (original DC gain = 1.1):"
puts sys_r.dcgain

# 17. Iterative LQR (iLQR)
puts "\n17. Iterative LQR:"
f_nonlin = ->(x : Float64Tensor, u : Float64Tensor) {
  a = [[1.0, 0.1], [0.0, 1.0]].to_tensor
  b = [[0.0], [0.1]].to_tensor
  a.matmul(x) + b.matmul(u)
}
jf_nonlin = ->(x : Float64Tensor, u : Float64Tensor) { [[1.0, 0.1], [0.0, 1.0]].to_tensor }
ju_nonlin = ->(x : Float64Tensor, u : Float64Tensor) { [[0.0], [0.1]].to_tensor }
xs, us = CrySpace::AdaptiveNonlinear.ilqr(
  f_nonlin, jf_nonlin, ju_nonlin,
  [[1.0, 0.0], [0.0, 1.0]].to_tensor,
  [[0.1]].to_tensor,
  [[10.0, 0.0], [0.0, 10.0]].to_tensor,
  [[1.0], [0.0]].to_tensor,
  [ [[0.0]].to_tensor, [[0.0]].to_tensor ]
)
puts "iLQR optimized final state:"
puts xs.last

# 18. Model Reference Adaptive Control (MRAC)
puts "\n18. Model Reference Adaptive Control (MRAC):"
r_sig = ->(t : Float64) { 1.0 }
t_vec = Float64Tensor.linear_space(0.0, 1.0, 11)
xs_mrac, xms_mrac, us_mrac = CrySpace::AdaptiveNonlinear.mrac_simulate(-1.0, 2.0, -2.0, 2.0, 1.0, 1.0, r_sig, t_vec)
puts "MRAC tracking output trajectory at t=1.0s: #{xs_mrac[-1].value}"

# 19. Backlash Describing Function
puts "\n19. Backlash Describing Function:"
df_backlash = CrySpace::AdaptiveNonlinear.describing_function_backlash(2.0, 0.5)
puts "Describing function complex gain for amp=2.0, half-width=0.5:"
puts df_backlash
