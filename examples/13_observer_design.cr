# 13_observer_design.cr
# This example demonstrates full-state observer design via pole placement (acker_obs).
# An observer estimates the internal state variables x(t) of a system using only inputs u(t) 
# and output measurements y(t).

require "../src/cryspace"

puts "=== 1. DEFINE SYSTEM MODEL ==="
# Double integrator plant (A = [0 1; 0 0], B = [0; 1], C = [1 0])
# States: x1 = position, x2 = velocity
a = [[0.0, 1.0], [0.0, 0.0]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys

# Verify system is observable (so we can reconstruct states from output y)
puts "Is observable? #{sys.is_observable?}"

puts "\n=== 2. DESIGN FULL-STATE OBSERVER ==="
# We want to place the observer poles (A - L*C) to be faster than the plant's poles.
# Placing observer poles at -4.0 and -5.0.
obs_poles = [-4.0, -5.0]
l = sys.acker_obs(obs_poles)

puts "Computed Observer Gain Matrix L (n x 1):"
puts l

# Let's verify the observer eigenvalues (A - L*C)
a_obs = sys.a - l.matmul(sys.c)
obs_eigen = a_obs.eigvals_c.to_a

puts "Actual observer eigenvalues:"
obs_eigen.each do |p|
  puts " - #{p.real.round(4)} + #{p.imag.round(4)}i"
end
