# 12_discrete_lqr.cr
# This example demonstrates optimal discrete-time control design using DLQR (Discrete Linear Quadratic Regulator).
# It defines a discrete-time unstable system, solves the Discrete Algebraic Riccati Equation (DARE),
# computes the optimal state-feedback gain matrix K, and verifies stability of the closed loop.

require "../src/cryspace"

puts "=== 1. DEFINE DISCRETE-TIME SYSTEM ==="
# A discrete-time system x[k+1] = A*x[k] + B*u[k]
# A = [[1.5, 1.0], [0.0, 0.8]] (unstable since eigenvalue 1.5 is outside the unit circle |z| < 1)
# B = [[0.0], [1.0]]
dt = 0.1
a = [[1.5, 1.0], [0.0, 0.8]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys_d = CrySpace::StateSpace.new(a, b, c, d, dt)
puts sys_d
puts "Open-loop discrete poles: #{sys_d.poles.map { |p| "#{p.real.round(2)} + #{p.imag.round(2)}i" }}"
puts "Is open-loop stable? #{sys_d.is_stable?}"

puts "\n=== 2. DESIGN DLQR CONTROLLER ==="
# Weighting matrices Q (state cost) and R (control cost)
q = [[10.0, 0.0], [0.0, 1.0]].to_tensor
r = [[1.0]].to_tensor

# Compute optimal feedback gain K, DARE solution P, and closed-loop poles
k, p, cl_poles = sys_d.dlqr(q, r)

puts "Optimal Gain Matrix K:"
puts k
puts "DARE Solution P:"
puts p
puts "Closed-loop discrete poles: #{cl_poles.map { |p| "#{p.real.round(4)} + #{p.imag.round(4)}i" }}"

# Verify closed-loop stability (all poles must have |z| < 1)
# A_cl = A - B*K
sys_cl = CrySpace::StateSpace.new(sys_d.a - sys_d.b.matmul(k), sys_d.b, sys_d.c, sys_d.d, dt)
puts "Is closed-loop stable? #{sys_cl.is_stable?}"
