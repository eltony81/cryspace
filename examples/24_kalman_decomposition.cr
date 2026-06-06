# 24_kalman_decomposition.cr
# This example demonstrates:
# 1. Similarity Transformation (similarity_transform)
# 2. Kalman Controllability Decomposition (controllable_decomposition)
# 3. Kalman Observability Decomposition (observable_decomposition)
# 4. Minimal Realization (minreal) to eliminate uncontrollable and unobservable states

require "../src/cryspace"

puts "=== 1. DEFINE SYSTEM WITH UNCONTROLLABLE & UNOBSERVABLE STATES ==="
# A 3-state system:
# x1: controllable and observable (pole at -1)
# x2: uncontrollable (pole at -2)
# x3: unobservable (pole at -3)
#
# dx/dt = [-1 0 0; 0 -2 0; 0 0 -3]*x + [1; 0; 1]*u
# y     = [1 1 0]*x
a = [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -3.0]].to_tensor
b = [[1.0], [0.0], [1.0]].to_tensor
c = [[1.0, 1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts "Original System states: #{sys.n_states}"
puts sys

puts "\n=== 2. SIMILARITY TRANSFORMATION ==="
# Scale states: z = T * x => T = diag(2, 2, 2)
t_matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]].to_tensor
sys_transformed = sys.similarity_transform(t_matrix)
puts "Transformed System (scaled by 2):"
puts sys_transformed

puts "\n=== 3. KALMAN CONTROLLABILITY DECOMPOSITION ==="
sys_c, t_c, r_c = sys.controllable_decomposition
puts "Controllable Rank: #{r_c}"
puts "Controllable Decomposition A matrix:"
puts sys_c.a

puts "\n=== 4. KALMAN OBSERVABILITY DECOMPOSITION ==="
sys_o, t_o, r_o = sys.observable_decomposition
puts "Observable Rank: #{r_o}"
puts "Observable Decomposition A matrix:"
puts sys_o.a

puts "\n=== 5. MINIMAL REALIZATION (MINREAL) ==="
sys_min = sys.minreal
puts "Minimal Realization states: #{sys_min.n_states} (Reduced from #{sys.n_states})"
puts sys_min
puts "Minimal Realization Poles:"
sys_min.poles.each do |p|
  puts "  #{p.real.round(3)} + #{p.imag.round(3)}i"
end
