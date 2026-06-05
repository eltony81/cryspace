# 15_lsim_simulation.cr
# This example demonstrates linear simulation using the lsim method.
# It simulates a second-order system forced by a time-varying input (e.g., a sine wave).

require "../src/cryspace"

puts "=== 1. DEFINE SYSTEM MODEL ==="
# A stable harmonic oscillator: G(s) = 10 / (s^2 + 2s + 10)
num = [10.0].to_tensor
den = [1.0, 2.0, 10.0].to_tensor
sys = CrySpace::TransferFunction.new(num, den).to_statespace
puts sys

puts "\n=== 2. GENERATE TIME-VARYING INPUT ==="
# Time range: 0 to 5 seconds (51 steps, dt = 0.1s)
t_vec = Float64Tensor.linear_space(0.0, 5.0, 51)

# Input: u(t) = sin(2 * pi * t)
u_vec = Float64Tensor.new([1, t_vec.size])
t_vec.size.times do |i|
  u_vec[0, i] = Math.sin(2.0 * Math::PI * t_vec[i].value)
end

puts "\n=== 3. SIMULATE SYSTEM RESPONSE USING LSIM ==="
t_out, x_out, y_out = sys.lsim(u_vec, t_vec, method: :rk4)

puts "Time (s) | Input u(t) | Output y(t)"
puts "------------------------------------"
t_out.size.times do |i|
  # Print every 5 steps to keep output brief
  if i % 5 == 0
    t_val = t_out[i].value
    u_val = u_vec[0, i].value
    y_val = y_out[i, 0].value
    puts "  #{t_val.round(1).to_s.ljust(6)} |   #{u_val.round(4).to_s.ljust(8)} |   #{y_val.round(4)}"
  end
end
