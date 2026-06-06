require "../src/cryspace"

puts "=================================================="
puts "CrySpace: Showcasing 9 Advanced Control Features"
puts "=================================================="

# 1. Sylvester Equation Solver
puts "\n1. Sylvester Equation Solver:"
a = [[-1.0, 0.0], [0.0, -2.0]].to_tensor
b = [[-3.0, 0.0], [0.0, -4.0]].to_tensor
c = [[1.0, 1.0], [1.0, 1.0]].to_tensor
x = CrySpace::StateSpace.sylvester(a, b, c)
puts "Solved X for A*X + X*B = C:"
puts x

# 2. Robust Pole Placement
puts "\n2. Robust Pole Placement (place):"
a_sys = [[0.0, 1.0], [-2.0, -3.0]].to_tensor
b_sys = [[0.0], [1.0]].to_tensor
c_sys = [[1.0, 0.0]].to_tensor
d_sys = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a_sys, b_sys, c_sys, d_sys)
k = sys.place([-5.0, -6.0])
puts "Designed State Feedback gain K to place poles at [-5.0, -6.0]:"
puts k

# 3. H2 Norm
puts "\n3. H2 Norm Calculation:"
sys_1st = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
puts "H2 norm of 1/(s+1) is (expected ~0.7071):"
puts sys_1st.h2norm

# 4. Matched Pole-Zero Discretization
puts "\n4. Matched Pole-Zero Discretization:"
g = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 2.0].to_tensor)
g_d = g.to_discrete(0.1, :matched)
puts "Continuous 1/(s+2) discretized using matched method (dt=0.1):"
puts g_d

# 5. Lead-Lag Compensator
puts "\n5. Lead-Lag Compensator Design:"
comp = CrySpace::TransferFunction.leadlag(2.0, 10.0, 1.5)
puts "Lead-lag Compensator (1.5 * (s+2)/(s+10)):"
puts comp

# 6. Loop Shaping Weight Generator
puts "\n6. Loop Shaping Weight Generator (makeweight):"
w = CrySpace::TransferFunction.makeweight(0.1, 10.0, 2.0)
puts "Loop shaping weighting filter W(s):"
puts w

# 7. Least Squares Parameter Estimation
puts "\n7. Least Squares System Identification:"
dt = 0.1
r = Random.new(42)
u = Float64Tensor.zeros([100])
100.times { |i| u[i] = r.rand(-1.0..1.0) }
y = Float64Tensor.zeros([100])
y[0] = 2.0 * u[0].value
(1..99).each do |k|
  y[k] = 0.5 * y[k-1].value + 2.0 * u[k].value
end
sys_ident = CrySpace::Ident.least_squares_estimation(u, y, 1, dt)
puts "Identified Discrete Transfer Function (expected den ~ [1, -0.5], num ~ [2]):"
puts sys_ident

# 8. Eigenvalue Realization Algorithm (ERA)
puts "\n8. ERA realization from impulse response:"
ir = [0.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625].to_tensor
sys_real = CrySpace::Ident.era(ir, 1, 1, 1, 0.1)
puts "Realized StateSpace system A matrix (expected ~0.5):"
puts sys_real.a

# 9. Describing Functions for Nonlinearities
puts "\n9. Describing Functions:"
sat_val = CrySpace::Nonlinear.describing_function_saturation(2.0, 1.0)
dead_val = CrySpace::Nonlinear.describing_function_deadzone(1.0, 0.5)
puts "Saturation describing function for amplitude 2.0, limit 1.0 (expected ~0.609):"
puts sat_val
puts "Deadzone describing function for amplitude 1.0, half-width 0.5 (expected ~0.391):"
puts dead_val
