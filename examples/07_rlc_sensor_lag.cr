# 07_rlc_sensor_lag.cr
# This example demonstrates a closed-loop control system with sensor dynamics (lag) in the feedback path.
# Rather than unity feedback, the measured output lags due to a first-order sensor low-pass filter.
# Adapted from the plotter_sensor app.

require "../src/cryspace"

# 1. Plant (RLC) - Standard parameters from UMich Tutorial
# Physical parameters: Resistance R = 1.0 Ohm, Inductance L = 0.5 H, Capacitance C = 0.1 F
R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)

# 2. PID Controller (Forward Path) - Optimized to minimize settling time under sensor lag
# These gains were determined using a multi-core grid search that minimized settling time
# (the time after which the output enters and stays within a +/-5% band of the target value).
# This prevents the output from dropping back down to ~0.6V and ensures rapid steady-state tracking.
# Optimization method details:
#   - Constraints: Plant and Sensor Overshoot <= 15%
#   - Score function: score = SettlingTime + 0.01 * (Overshoot_Plant + Overshoot_Sensor)
#   - Resulting gains: Kp = 1.0, Ki = 8.8, Kd = 0.5
kp, ki, kd = 1.0, 8.8, 0.5
tf = 0.01 # Derivative filter for realizability
pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
pid_den = [tf, 1.0, 0.0].to_tensor
pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace

# Forward Path System (Plant * Controller)
sys_forward = rlc_plant * pid_controller

# 3. Sensor Dynamics in the Feedback Path (Measurement lag)
# We model a sensor with a low-pass measurement lag (time constant tau = 0.05 seconds):
# H(s) = 1 / (0.05s + 1)
sensor_tau = 0.05
sensor_num = [1.0].to_tensor
sensor_den = [sensor_tau, 1.0].to_tensor
sensor_dynamics = CrySpace::TransferFunction.new(sensor_num, sensor_den).to_statespace

# 4. Closed-loop connection with sensor dynamics in feedback path
# Under negative feedback, y = sys_forward(u - sensor_dynamics(y))
sys_cl = sys_forward.feedback(sensor_dynamics)

# 5. Simulation
# Time vector: 0 to 15 seconds with dt = 0.01 seconds (1501 points)
t_vec = Float64Tensor.linear_space(0.0, 15.0, 1501)
u_step = Float64Tensor.ones([1, 1501])

# Simulate open loop and closed loop
_, _, y_open = rlc_plant.simulate(t_vec, u: u_step.dup)
_, _, y_cl = sys_cl.simulate(t_vec, u: u_step.dup)

# 6. Export results to CSV
csv_filename = "examples/results_sensor.csv"
File.open(csv_filename, "w") do |file|
  file.puts "Time,OpenLoop,ClosedLoopSensorPID"
  1501.times { |i| file.puts "#{t_vec[i].value},#{y_open[i, 0].value},#{y_cl[i, 0].value}" }
end
puts "Exported sensor-lag step response data to #{csv_filename}"

# Print sample results to verify convergence
puts "\nTime (s) | Open Loop Output | Closed Loop Output (with Sensor Lag)"
puts "------------------------------------------------------------------"
[0, 200, 400, 600, 800, 1000, 1200, 1400].each do |idx|
  puts "  #{t_vec[idx].value.round(2).to_s.ljust(6)} | #{y_open[idx, 0].value.round(5).to_s.ljust(16)} | #{y_cl[idx, 0].value.round(5)}"
end
