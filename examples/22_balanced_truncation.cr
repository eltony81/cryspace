# 22_balanced_truncation.cr
# This example demonstrates Balanced Realization and Balanced Truncation model reduction
# using the `balred` method. It takes a 4th-order active filter model and reduces it
# to a 2nd-order approximation while retaining the dominant dynamics.

require "../src/cryspace"

puts "=== 1. DEFINE A 4TH-ORDER ACTIVE FILTER PLANT ==="
# A 4-state stable linear system with poles at -1, -2, -8, -12
a = [
  [-1.0,  0.0,  0.0,   0.0],
  [ 0.0, -2.0,  0.0,   0.0],
  [ 0.0,  0.0, -8.0,   0.0],
  [ 0.0,  0.0,  0.0, -12.0]
].to_tensor

b = [[1.0], [1.0], [1.0], [1.0]].to_tensor
c = [[1.0, 1.0, 1.0, 1.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys
puts "Original poles: #{sys.poles.map { |p| p.real.round(2) }}"

puts "\n=== 2. COMPUTE HANKEL SINGULAR VALUES ==="
# Hankel singular values show the energy of each state (controllability/observability)
hsv = sys.hsvd
puts "Hankel Singular Values: #{hsv.map { |val| val.round(4) }}"

puts "\n=== 3. PERFORM BALANCED TRUNCATION (REDUCE TO ORDER 2) ==="
# We truncate the 2 least significant states (poles at -8 and -12)
sys_red = sys.balred(2)
puts sys_red
puts "Reduced poles: #{sys_red.poles.map { |p| p.real.round(2) }}"

puts "\n=== 4. COMPARE TIME RESPONSE ==="
# Run a step response simulation on both systems
t = Float64Tensor.linear_space(0.0, 4.0, 41)
u = Float64Tensor.ones([1, t.size])
_, _, y_orig = sys.simulate(t, u: u.dup)
_, _, y_red = sys_red.simulate(t, u: u.dup)

puts "Time (s) | Original Output | Reduced Output"
puts "-------------------------------------------"
[0, 10, 20, 30, 40].each do |idx|
  time_val = t[idx].value.round(2)
  orig_val = y_orig[idx, 0].value.round(4)
  red_val  = y_red[idx, 0].value.round(4)
  puts "  #{time_val.to_s.ljust(8)} | #{orig_val.to_s.ljust(15)} | #{red_val}"
end
