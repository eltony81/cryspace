# 18_root_locus.cr
# This example demonstrates root locus calculations.
# It computes the trajectories of the closed-loop poles of a feedback system
# as the open-loop gain K increases from 0 to 10.

require "../src/cryspace"

puts "=== DEFINE OPEN LOOP SYSTEM ==="
# Open-loop system: L(s) = 1 / (s^2 + 2s) [Double integrator with a damping pole]
# A = [[0, 1], [0, -2]]
a = [[0.0, 1.0], [0.0, -2.0]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys

puts "\n=== COMPUTE ROOT LOCUS TRAJECTORIES ==="
# Gains to evaluate: 0.0, 0.5, 1.0, 2.0, 5.0, 10.0
gains = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0].to_tensor
poles_matrix = sys.root_locus(gains)

puts "Gain (K) | Closed-Loop Poles"
puts "----------------------------------------"
gains.size.times do |i|
  k = gains.to_unsafe[i]
  poles = poles_matrix[i].map { |p| "#{p.real.round(3)} + #{p.imag.round(3)}i" }.join(", ")
  puts "  #{k.to_s.ljust(6)} | [#{poles}]"
end
