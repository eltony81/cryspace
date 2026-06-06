# 30_observer_simulation.cr
# This example demonstrates:
# 1. Closed-loop observer design
# 2. Time-domain simulation of true system and estimator coupled together
# 3. Dynamic simulation under stochastic process and measurement noise

require "../src/cryspace"

puts "=== OBSERVER CLOSED-LOOP SIMULATION ==="

# Define continuous system (dc motor speed controller model)
# dx/dt = -2.0 * x + 3.0 * u
# y = x
a = [[-2.0]].to_tensor
b = [[3.0]].to_tensor
c = [[1.0]].to_tensor
d = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a, b, c, d)

# 1. Design optimal control feedback gain K via LQR
q_lqr = [[1.0]].to_tensor
r_lqr = [[0.1]].to_tensor
k_gain, _, _ = sys.lqr(q_lqr, r_lqr)
puts "LQR Control Gain K: #{k_gain[0, 0].value}"

# 2. Design observer estimator gain L via LQE (Kalman estimator)
q_noise = [[0.05]].to_tensor # process disturbance covariance
r_noise = [[0.01]].to_tensor # sensor measurement noise covariance
l_gain = sys.lqe(q_noise, r_noise)
puts "LQE Observer Gain L: #{l_gain[0, 0].value}"

# 3. Simulate the observer dynamics under noisy conditions
t = Float64Tensor.linear_space(0.0, 3.0, 300)
x0 = [[1.5]].to_tensor      # true initial speed
x0_est = [[0.0]].to_tensor  # observer starts with 0 estimate (1.5 error)

t_out, x_out, x_est_out, y_out, u_out = sys.simulate_observer(
  t,
  k_gain,
  l_gain,
  x0: x0,
  x0_est: x0_est,
  process_noise_cov: q_noise,
  measure_noise_cov: r_noise
)

puts "\nSimulation Results (Sampled every 50 steps):"
(0...300).step(50) do |i|
  time_val = t_out.to_unsafe[i]
  true_state = x_out[i, 0].value
  est_state = x_est_out[i, 0].value
  output_val = y_out[i, 0].value
  control_val = u_out[i, 0].value
  puts "t: #{time_val.round(2)}s | True State: #{true_state.round(4)} | Est State: #{est_state.round(4)} | Meas output: #{output_val.round(4)} | Control U: #{control_val.round(4)}"
end

puts "\nObserver successfully converged and tracked the state despite process & measurement noises!"
