# 01_statespace_basic.cr
# This example demonstrates the basic capabilities of the StateSpace class in CrySpace.
# It defines a continuous-time state-space representation, displays properties,
# calculates poles, DC gain, stability, and performs controllability/observability analysis.

require "../src/cryspace"

# Let's define a second-order spring-mass-damper system (an oscillator):
# m * y'' + c * y' + k * y = u
# Mass (m) = 1.0, Stiffness (k) = 10.0, Damping (c) = 0.5
m, k, c = 1.0, 10.0, 0.5

# State-space formulation: x' = A*x + B*u, y = C*x + D*u
# States: x1 = position (y), x2 = velocity (y')
a = [[0.0, 1.0], [-k/m, -c/m]].to_tensor
b = [[0.0], [1/m]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

puts "=== 1. CREATING A STATESPACE SYSTEM ==="
sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys
puts "Number of states:  #{sys.n_states}"
puts "Number of inputs:  #{sys.n_inputs}"
puts "Number of outputs: #{sys.n_outputs}"

puts "\n=== 2. POLES AND DC GAIN ==="
# Calculate poles (eigenvalues of the system matrix A)
poles = sys.poles
puts "System Poles: #{poles.map { |p| "#{p.real.round(4)} + #{p.imag.round(4)}i" }}"

# Calculate the DC Gain (steady-state gain)
# For continuous systems: G(0) = D - C * A^-1 * B
dcgain = sys.dcgain
puts "DC Gain (steady-state): #{dcgain[0, 0].value.round(4)}"

puts "\n=== 3. STABILITY AND SYSTEM PROPERTIES ==="
# Check if the system is stable (all poles have real parts < 0 for continuous-time systems)
puts "Is the system stable? #{sys.is_stable?}"

# Controllability matrix: Co = [B, AB, A^2B, ... A^(n-1)B]
ctrb_matrix = sys.ctrb
puts "Controllability Matrix:"
puts ctrb_matrix

# Observability matrix: Ob = [C; CA; CA^2; ... CA^(n-1)]
obsv_matrix = sys.obsv
puts "Observability Matrix:"
puts obsv_matrix

# Check if full-rank / fully controllable and observable
puts "Is controllable? #{sys.is_controllable?}"
puts "Is observable?   #{sys.is_observable?}"
