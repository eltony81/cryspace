# 09_lyapunov_stability.cr
# This example demonstrates solving Lyapunov equations for both continuous and discrete systems.
# Lyapunov equations are fundamental to verifying stability and computing system gramians.

require "../src/cryspace"

# Let's define a stable continuous-time system matrix A:
# A = [[-2, 1], [-1, -3]]
a_cont = [[-2.0, 1.0], [-1.0, -3.0]].to_tensor
b_dummy = [[0.0], [0.0]].to_tensor
c_dummy = [[1.0, 1.0]].to_tensor
d_dummy = [[0.0]].to_tensor
sys_cont = CrySpace::StateSpace.new(a_cont, b_dummy, c_dummy, d_dummy)

puts "=== 1. CONTINUOUS-TIME LYAPUNOV EQUATION ==="
# Solve: A*P + P*A^T + Q = 0
# We specify a symmetric positive-definite matrix Q:
q_cont = [[1.0, 0.0], [0.0, 1.0]].to_tensor

p_cont = sys_cont.lyap(q_cont)
puts "Solution P:"
puts p_cont

# Verify the result: A*P + P*A^T should equal -Q
verification = sys_cont.a.matmul(p_cont) + p_cont.matmul(sys_cont.a.transpose)
puts "Verification (A*P + P*A^T):"
puts verification

puts "\n=== 2. DISCRETE-TIME LYAPUNOV EQUATION ==="
# Solve: A_d*P*A_d^T - P + Q_d = 0
# Let's discretize our system first:
sys_disc = sys_cont.sample(0.1)

q_disc = [[1.0, 0.0], [0.0, 1.0]].to_tensor
p_disc = sys_disc.dlyap(q_disc)
puts "Discrete Solution P:"
puts p_disc

# Verify result: A*P*A^T - P should equal -Q
verification_disc = sys_disc.a.matmul(p_disc).matmul(sys_disc.a.transpose) - p_disc
puts "Verification (A*P*A^T - P):"
puts verification_disc
