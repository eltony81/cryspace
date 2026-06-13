# 06_rlc_pid_control.cr
# This example demonstrates a closed-loop control system:
# An RLC plant circuit controlled by a continuous-time PID controller (with a derivative filter)
# under unity negative feedback. This is adapted from the plotter app.

require "../src/cryspace"

# 1. Plant (RLC) - Standard parameters from UMich Tutorial
# Physical parameters: Resistance R = 1.0 Ohm, Inductance L = 0.5 H, Capacitance C = 0.1 F
R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)

# 2. PID Controller - Optimized for fast settling time (0.08s) and zero overshoot.
# These gains were determined using a multi-core grid search that minimized settling time
# (the time after which the output enters and stays within a +/-5% band of the target value).
# Optimization method details:
#   - Constraints: Overshoot <= 15%
#   - Score function: score = SettlingTime + 0.01 * Overshoot
#   - Resulting gains: Kp = 2.2, Ki = 14.6, Kd = 1.0
kp, ki, kd = 2.2, 14.6, 1.0
tf = 0.01 # Derivative filter for stability
pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
pid_den = [tf, 1.0, 0.0].to_tensor
pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace

# 3. Closed-loop connection
# Unity negative feedback loop: G_cl = Forward / (1 + Forward)
# Feedback factor H = 1.0, represented as a 1x1 matrix tensor.
sys_cl = (rlc_plant * pid_controller).feedback([[1.0]].to_tensor)

# 4. Simulation
# Time vector: 0 to 15 seconds with dt = 0.01 seconds (1501 points)
t_vec = Float64Tensor.linear_space(0.0, 15.0, 1501)
u_step = Float64Tensor.ones([1, 1501])

# Simulate open loop and closed loop
_, _, y_open = rlc_plant.simulate(t_vec, u: u_step.dup)
_, _, y_cl = sys_cl.simulate(t_vec, u: u_step.dup)

# 5. Export results to CSV or Feather (if Arrow is enabled)
{% if flag?(:arrow) %}
  arrow_t = Arrow::DoubleArray.new(t_vec.to_a)
  arrow_open = Arrow::DoubleArray.new(y_open[..., 0].to_a)
  arrow_cl = Arrow::DoubleArray.new(y_cl[..., 0].to_a)
  schema = Arrow::Schema.new([
    Arrow::Field.new("Time", Arrow::DataType.double),
    Arrow::Field.new("OpenLoop", Arrow::DataType.double),
    Arrow::Field.new("ClosedLoopPID", Arrow::DataType.double)
  ])
  table = Arrow::Table.new(schema, [arrow_t, arrow_open, arrow_cl])
  feather_filename = "examples/results.feather"
  writer = Arrow::FeatherWriter.new(feather_filename)
  writer.write(table)
  writer.close
  puts "Exported step response data to #{feather_filename} (Apache Arrow Feather format)"
  File.delete(feather_filename) if File.exists?(feather_filename) # Cleanup in examples run
{% else %}
  csv_filename = "examples/results.csv"
  File.open(csv_filename, "w") do |file|
    file.puts "Time,OpenLoop,ClosedLoopPID"
    1501.times { |i| file.puts "#{t_vec[i].value},#{y_open[i, 0].value},#{y_cl[i, 0].value}" }
  end
  puts "Exported step response data to #{csv_filename}"
{% end %}

# Print sample results to verify convergence
puts "\nTime (s) | Open Loop Output | Closed Loop Output"
puts "------------------------------------------------"
[0, 200, 400, 600, 800, 1000, 1200, 1400].each do |idx|
  puts "  #{t_vec[idx].value.round(2).to_s.ljust(6)} | #{y_open[idx, 0].value.round(5).to_s.ljust(16)} | #{y_cl[idx, 0].value.round(5)}"
end
