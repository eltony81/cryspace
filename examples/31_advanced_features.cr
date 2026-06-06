# 31_advanced_features.cr
# This example demonstrates:
# 1. Package-level block connections (series, parallel, feedback, append)
# 2. Butterworth lowpass filter design (TransferFunction.butter)
# 3. LQR Prefilter feedforward scaling gain (nbar)
# 4. Ramp response simulation (ramp_response)
# 5. MIMO Singular Value Analysis (sigma_data)

require "../src/cryspace"

puts "=== 1. PACKAGE-LEVEL BLOCK CONNECTIONS ==="
sys1 = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
sys2 = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)

# Append (combines systems in block-diagonal parallel form)
sys_appended = CrySpace.append(sys1, sys2)
puts "Appended System A matrix (should be diagonal block [ -1 0; 0 -2 ]):"
puts sys_appended.a

# Series, Parallel, and Feedback helpers
sys_s = CrySpace.series(sys1, sys2)
sys_p = CrySpace.parallel(sys1, sys2)
sys_f = CrySpace.feedback(sys1, sys2)
puts "Module connections successfully computed: series (#{sys_s.n_states} states), parallel (#{sys_p.n_states} states), feedback (#{sys_f.n_states} states)."

puts "\n=== 2. ANALOG BUTTERWORTH LOWPASS FILTER ==="
# 2nd-order Butterworth filter with cutoff Wn = 5.0 rad/s
# Denominator should be s^2 + sqrt(2)*Wn*s + Wn^2 = s^2 + 7.071*s + 25.0
butter_tf = CrySpace::TransferFunction.butter(2, 5.0)
puts "Designed 2nd-order Butterworth Filter (cutoff 5 rad/s):"
puts butter_tf

puts "\n=== 3. LQR PREFILTER FEEDFORWARD SCALING GAIN (nbar) ==="
# Plant model: dx/dt = -3.0*x + 2.0*u, y = x
plant = CrySpace::StateSpace.new([[-3.0]].to_tensor, [[2.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
k_gain = [[1.5]].to_tensor # LQR feedback control gain

# Prefilter gain N to ensure steady state output y = reference r
n_scale = plant.nbar(k_gain)
puts "Prefilter Tracking Gain N (should be 3.0):"
puts n_scale

puts "\n=== 4. RAMP RESPONSE SIMULATION ==="
# Simulate system tracking ramp input
t_ramp, x_ramp, y_ramp = plant.ramp_response(n_steps: 10)
puts "Ramp response outputs (starts at 0.0, increases):"
y_ramp.each_with_index do |y_val, idx|
  puts "  t = #{t_ramp[idx].round(2)}s | Output y = #{y_val[0, 0].value.round(4)}"
end

puts "\n=== 5. MIMO SINGULAR VALUE ANALYSIS (sigma_data) ==="
# Define a simple 2x2 MIMO system
a_mimo = [[-1.0, 0.0], [0.0, -2.0]].to_tensor
b_mimo = [[1.0, 0.0], [0.0, 1.0]].to_tensor
c_mimo = [[1.0, 1.0], [0.0, 1.0]].to_tensor
d_mimo = [[0.0, 0.0], [0.0, 0.0]].to_tensor
sys_mimo = CrySpace::StateSpace.new(a_mimo, b_mimo, c_mimo, d_mimo)

# Evaluate singular values over 3 frequency points (w = 0.1, 1.0, 10.0 rad/s)
omega = [0.1, 1.0, 10.0].to_tensor
_, sv = sys_mimo.sigma_data(omega)

puts "MIMO Singular Values at w = 0.1, 1.0, 10.0 rad/s:"
3.times do |i|
  puts "  w = #{omega[i].value} rad/s | Singular values: [#{sv[i, 0].value.round(4)}, #{sv[i, 1].value.round(4)}]"
end
