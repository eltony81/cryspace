# 26_advanced_realizations.cr
# This example demonstrates:
# 1. State-Space realizations (to_control_canonical_form, to_observable_canonical_form)
# 2. Impulse Response (impulse_response) and Initial State Free Response (initial_response)
# 3. System Bandwidth calculation (bandwidth)

require "../src/cryspace"

puts "=== 1. DEFINE A SECOND-ORDER SISO SYSTEM ==="
# G(s) = 1 / (s^2 + 3s + 2)
# dx/dt = [0 1; -2 -3]*x + [0; 1]*u
# y     = [1 0]*x
a = [[0.0, 1.0], [-2.0, -3.0]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts sys
puts "DC Gain: #{sys.dcgain[0, 0].value}"

puts "\n=== 2. TRANSFORMATION TO CANONICAL FORMS ==="
# Control Canonical Form
sys_ccf, t_ccf = sys.to_control_canonical_form
puts "Control Canonical Form (CCF) A matrix:"
puts sys_ccf.a
puts "Transformation matrix (T_ccf):"
puts t_ccf

# Observable Canonical Form
sys_ocf, t_ocf = sys.to_observable_canonical_form
puts "\nObservable Canonical Form (OCF) A matrix:"
puts sys_ocf.a

puts "\n=== 3. COMPUTE SYSTEM BANDWIDTH ==="
# Bandwidth is the frequency where the gain magnitude drops by -3dB (i.e. to 0.707 of DC Gain)
w_bw = sys.bandwidth
puts "System Bandwidth: #{w_bw.round(4)} rad/s"

puts "\n=== 4. SIMULATE IMPULSE AND INITIAL RESPONSES ==="
# Impulse response
t_imp, x_imp, y_imp = sys.impulse_response(n_steps: 10)
puts "Impulse response (first 5 steps):"
5.times do |i|
  puts "  t = #{t_imp[i].round(2)}s, Output y = #{y_imp[i][0, 0].value.round(4)}"
end

# Free Response from non-zero initial state x0 = [1.0, -1.0]
x0 = [[1.0], [-1.0]].to_tensor
t_init, x_init, y_init = sys.initial_response(x0, n_steps: 10)
puts "\nInitial response from x0 = [1.0; -1.0] (first 5 steps):"
5.times do |i|
  puts "  t = #{t_init[i].round(2)}s, Output y = #{y_init[i][0, 0].value.round(4)}"
end
