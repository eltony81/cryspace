# 23_lqg_design.cr
# This example demonstrates the design of a Linear Quadratic Gaussian (LQG) controller.
# It designs an LQR state feedback gain (K) and a Kalman Filter estimator gain (L),
# and combines them into a single StateSpace LQG controller.

require "../src/cryspace"

puts "=== 1. DEFINE PLANT MODEL ==="
# A continuous double integrator system (e.g. tracking position and velocity)
# dx/dt = [0 1; 0 0]*x + [0; 1]*u
# y     = [1 0]*x
a = [[0.0, 1.0], [0.0, 0.0]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys

puts "\n=== 2. DESIGN LQR GAIN (K) ==="
# Weighting matrices for LQR state feedback
q_lqr = [[10.0, 0.0], [0.0, 1.0]].to_tensor
r_lqr = [[1.0]].to_tensor
k_gain, _, _ = sys.lqr(q_lqr, r_lqr)
puts "LQR Gain (K): #{k_gain}"

puts "\n=== 3. DESIGN KALMAN FILTER GAIN (L) ==="
# Process and measurement noise covariance matrices for observer pole placement
# Using observer placement (acker_obs) to design estimator poles at -8, -10
l_gain = sys.acker_obs([-8.0, -10.0])
puts "Estimator Gain (L): #{l_gain}"

puts "\n=== 4. SYNTHESIZE LQG CONTROLLER ==="
# Synthesize the LQG controller which takes measured output y and produces control input u
controller_lqg = sys.lqg(k_gain, l_gain)
puts controller_lqg

puts "\n=== 5. ANALYZE CLOSED LOOP SYSTEM ==="
# The closed loop state-space equations with LQG controller:
# dx/dt     = A*x + B*u   = A*x - B*K*x_hat
# dx_hat/dt = (A - B*K - L*C)*x_hat + L*y
# We verify the closed-loop poles are stable.
sys_cl = sys.feedback(controller_lqg)
puts "Closed Loop Poles:"
sys_cl.poles.each do |p|
  puts "  #{p.real.round(3)} + #{p.imag.round(3)}i"
end
puts "Closed loop is stable? #{sys_cl.is_stable?}"
