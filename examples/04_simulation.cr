# 04_simulation.cr
# This example demonstrates time-domain simulation in CrySpace.
# It covers standard responses (step and impulse) and custom vectorized simulation
# using different integration methods (Runge-Kutta 4 and Euler).

require "../src/cryspace"

# Define a stable second-order system (underdamped harmonic oscillator)
# G(s) = 5 / (s^2 + s + 5)
num = [5.0].to_tensor
den = [1.0, 1.0, 5.0].to_tensor
sys = CrySpace::TransferFunction.new(num, den).to_statespace

puts "=== 1. STANDARD STEP RESPONSE ==="
# Simulates the step response using discrete-time approximation
# Returns: {time_array, states_array, output_array}
t_step, x_step, y_step = sys.step_response(n_steps: 10)
t_step.size.times do |i|
  puts "t = #{t_step[i].round(2)}s | Output = #{y_step[i][0, 0].value.round(4)}"
end

puts "\n=== 2. STANDARD IMPULSE RESPONSE ==="
# Simulates the impulse response (approximate delta function input)
t_imp, x_imp, y_imp = sys.impulse_response(n_steps: 5)
t_imp.size.times do |i|
  puts "t = #{t_imp[i].round(2)}s | Output = #{y_imp[i][0, 0].value.round(4)}"
end

puts "\n=== 3. CUSTOM VECTORIZED SIMULATION (Runge-Kutta 4) ==="
# We define a custom time grid and input signal (e.g., unit step)
t_vec = Float64Tensor.linear_space(0.0, 5.0, 6) # 6 steps from 0 to 5s
u_step = Float64Tensor.ones([sys.n_inputs, t_vec.size]) # input = 1.0 at all times

# Simulate using the RK4 method (Runge-Kutta 4th order, default)
t_out, x_out, y_out = sys.simulate(t_vec, u: u_step, method: :rk4)
t_out.size.times do |i|
  puts "t = #{t_out[i].value.round(2)}s | RK4 Output = #{y_out[i, 0].value.round(4)}"
end

puts "\n=== 4. SIMULATION WITH EULER METHOD ==="
# Simulate the exact same system using Euler integration
_, _, y_euler = sys.simulate(t_vec, u: u_step, method: :euler)
t_out.size.times do |i|
  puts "t = #{t_out[i].value.round(2)}s | Euler Output = #{y_euler[i, 0].value.round(4)}"
end
