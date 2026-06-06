# 21_d2c_conversion.cr
# This example demonstrates Discrete-to-Continuous time conversion (D2C)
# using the matrix logarithm state-space method.
# It discretizes a continuous system and then reconstructs it back to continuous time.

require "../src/cryspace"

puts "=== 1. ORIGINAL CONTINUOUS SYSTEM ==="
# Mass-damper system: x' = -2*x + u
sys_c = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
puts sys_c

puts "\n=== 2. DISCRETIZED SYSTEM (C2D) ==="
# Sample using dt = 0.5s
dt = 0.5
sys_d = sys_c.sample(dt)
puts sys_d
puts "Discrete pole: #{sys_d.poles[0].real.round(4)}"

puts "\n=== 3. RECONSTRUCTED CONTINUOUS SYSTEM (D2C) ==="
# Convert discrete system back to continuous time using matrix logarithm
sys_back = sys_d.to_continuous
puts sys_back
puts "Reconstructed continuous pole: #{sys_back.poles[0].real.round(4)}"
puts "Reconstructed matrix B:         #{sys_back.b[0, 0].value.round(4)}"
