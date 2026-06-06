# 32_additional_control_features.cr
# This example demonstrates:
# 1. StateSpace identity factories (StateSpace.eye)
# 2. Sensitivity (S) & Complementary Sensitivity (T) transfer functions
# 3. Filter transformations (lowpass_to_highpass, lowpass_to_bandpass)
# 4. Direct TF discretization (to_discrete) & continuous recovery (to_continuous)
# 5. MIMO Transmission zeros (transmission_zeros)
# 6. Balanced realizations (balreal) returning similarity transform matrices
# 7. LQR and DLQR controller synthesis with cross-coupling cost matrices (N_cross)

require "../src/cryspace"

puts "=== 1. IDENTITY STATE-SPACE SYSTEM ==="
sys_eye = CrySpace::StateSpace.eye(2)
puts "Identity StateSpace matrix D:"
puts sys_eye.d

puts "\n=== 2. SENSITIVITY (S) & COMPLEMENTARY SENSITIVITY (T) ==="
# Plant G(s) = 1 / (s + 1), Controller K(s) = 2
g = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 1.0].to_tensor)
k = CrySpace::TransferFunction.new([2.0].to_tensor, [1.0].to_tensor)

s = g.sensitivity(k)             # S = (s + 1) / (s + 3)
t_sens = g.complementary_sensitivity(k)  # T = 2 / (s + 3)

puts "Sensitivity Function S(s):"
puts s
puts "Complementary Sensitivity Function T(s):"
puts t_sens

puts "\n=== 3. FILTER FREQUENCY TRANSFORMATIONS ==="
# Start with normalized 1st-order Butterworth lowpass: G(s) = 1 / (s + 1)
lp_filter = CrySpace::TransferFunction.butter(1, 1.0)

# Lowpass to Highpass with cutoff at 3.0 rad/s: G_hp(s) = s / (s + 3)
hp_filter = lp_filter.lowpass_to_highpass(3.0)
puts "Transformed Highpass Filter:"
puts hp_filter

# Lowpass to Bandpass with center frequency 2.0 rad/s and bandwidth 0.5 rad/s
bp_filter = lp_filter.lowpass_to_bandpass(center_frequency: 2.0, bandwidth: 0.5)
puts "Transformed Bandpass Filter:"
puts bp_filter

puts "\n=== 4. DIRECT TRANSFER FUNCTION DISCRETIZATION & CONTINUOUS RECOVERY ==="
g_discrete = g.to_discrete(dt: 0.1, method: :zoh)
puts "Discretized Transfer Function (dt = 0.1):"
puts g_discrete

g_continuous = g_discrete.to_continuous
puts "Recovered Continuous Transfer Function:"
puts g_continuous

puts "\n=== 5. TRANSMISSION ZEROS ==="
# G(s) = (s + 4) / (s + 3) => Transmission zero at s = -4
sys_zero = CrySpace::StateSpace.new([[-3.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor)
zeros = sys_zero.transmission_zeros
puts "Transmission Zeros: #{zeros}"

puts "\n=== 6. BALANCED REALIZATION (balreal) ==="
# System: dx/dt = [ -1 0; 0 -2 ]*x + [1; 1]*u, y = [1 1]*x
sys_unbalanced = CrySpace::StateSpace.new(
  [[-1.0, 0.0], [0.0, -2.0]].to_tensor,
  [[1.0], [1.0]].to_tensor,
  [[1.0, 1.0]].to_tensor,
  [[0.0]].to_tensor
)
sys_balanced, t_mat, t_inv = sys_unbalanced.balreal
puts "Balanced Realization A Matrix:"
puts sys_balanced.a
puts "Balanced Realization Controllability Gramian (should be diagonal and equal to Observability):"
puts sys_balanced.gram(:c)

puts "\n=== 7. LQR WITH CROSS-COUPLING TERM ==="
# plant model: dx/dt = -3*x + 2*u
plant = CrySpace::StateSpace.new([[-3.0]].to_tensor, [[2.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
q = [[1.0]].to_tensor
r = [[1.0]].to_tensor
n_cross = [[0.5]].to_tensor

k_lqr, _, _ = plant.lqr(q, r, n_cross)
puts "LQR Gain with Cross-Coupling:"
puts k_lqr
