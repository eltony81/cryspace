# 08_lqr_control.cr
# This example demonstrates optimal state-feedback control design using LQR (Linear Quadratic Regulator).
# It defines an unstable second-order system, computes the LQR gain matrix, and verifies that the 
# closed-loop system becomes stable.

require "../src/cryspace"

puts "=== 1. DEFINE UNSTABLE SYSTEM ==="
# An unstable system (inverted pendulum-like system, where pole is in RHP)
# x' = A*x + B*u
# A = [[0, 1], [4, 0]]   (poles at +2 and -2, unstable)
# B = [[0], [1]]
a = [[0.0, 1.0], [4.0, 0.0]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys
puts "Open-loop poles: #{sys.poles.map { |p| "#{p.real.round(2)} + #{p.imag.round(2)}i" }}"
puts "Is open-loop stable? #{sys.is_stable?}"

puts "\n=== 2. DESIGN LQR CONTROLLER ==="
# Weighting matrices Q (state cost) and R (control cost)
# We penalize state deviations heavily and allow moderate control effort:
q = [[10.0, 0.0], [0.0, 1.0]].to_tensor
r = [[0.1]].to_tensor

# Compute optimal feedback gain matrix K and Riccati solution P
# Optimal law: u = -K*x
k, p, cl_poles = sys.lqr(q, r)

puts "Optimal Gain Matrix K:"
puts k
puts "Riccati Equation Solution P:"
puts p
puts "Closed-loop poles: #{cl_poles.map { |p| "#{p.real.round(4)} + #{p.imag.round(4)}i" }}"

# Verify stability of closed-loop system
# A_cl = A - B*K
sys_cl = CrySpace::StateSpace.new(sys.a - sys.b.matmul(k), sys.b, sys.c, sys.d)
puts "Is closed-loop stable? #{sys_cl.is_stable?}"
