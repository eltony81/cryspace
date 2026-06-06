# 25_lqe_design.cr
# This example demonstrates:
# 1. Optimal Kalman Estimator Design (lqe) for continuous systems
# 2. Discrete Optimal Kalman Estimator Design (dlqe) for discrete systems
# 3. Augmented StateSpace model with integrator (augment_integrator) for zero steady-state error tracking

require "../src/cryspace"

puts "=== 1. DEFINE CONTINUOUS PLANT MODEL ==="
# A stable first-order system: G(s) = 1 / (s + 1)
# dx/dt = -x + u
# y     = x
a = [[-1.0]].to_tensor
b = [[1.0]].to_tensor
c = [[1.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys

puts "\n=== 2. DESIGN OPTIMAL CONTINUOUS ESTIMATOR GAIN (LQE) ==="
# Q: Process noise covariance matrix
# R: Measurement noise covariance matrix
q_noise = [[1.0]].to_tensor
r_noise = [[1.0]].to_tensor

l_gain = sys.lqe(q_noise, r_noise)
puts "Continuous Estimator Gain (L):"
puts l_gain

puts "\n=== 3. DESIGN OPTIMAL DISCRETE ESTIMATOR GAIN (DLQE) ==="
# Discretize the system first with a sample time of 0.5s
dt = 0.5
sys_d = sys.sample(dt)
puts "Discretized System (dt = #{dt}):"
puts sys_d

l_gain_d = sys_d.dlqe(q_noise, r_noise)
puts "Discrete Estimator Gain (L_d):"
puts l_gain_d

puts "\n=== 4. AUGMENT SYSTEM WITH INTEGRATOR FOR TRACKING ==="
# Build augmented system to include integral of tracking error (e = r - y)
# Augmented state: [x; x_int]
# dx_aug/dt = [A 0; -C 0]*[x; x_int] + [B; -D]*u
sys_aug = sys.augment_integrator
puts "Augmented System states: #{sys_aug.n_states} (1 plant state + 1 integrator state)"
puts sys_aug

# Design a state-feedback controller for the augmented system
# Augmented states: [state, integrator_error]
q_lqr = [[1.0, 0.0], [0.0, 10.0]].to_tensor
r_lqr = [[1.0]].to_tensor

k_aug, _, _ = sys_aug.lqr(q_lqr, r_lqr)
puts "\nAugmented State Feedback Gain (K_aug):"
puts k_aug
puts "Where K_aug = [K_state, K_integrator]"
