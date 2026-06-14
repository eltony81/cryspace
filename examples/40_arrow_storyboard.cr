# examples/40_arrow_storyboard.cr
# This is the Apache Arrow accelerated version of the active vehicle suspension storyboard.
# It leverages the Arrow backend for simulations and writes data using Arrow::Table and FeatherWriter.

require "../src/cryspace"

puts "================================================================================"
puts "  CrySpace Apache Arrow Storyboard Tutorial"
puts "  Case Study: Active Vehicle Suspension Control System Design (Arrow SIMD)"
puts "================================================================================"

{% if flag?(:arrow) %}
  puts "Arrow support enabled!"
{% else %}
  puts "WARNING: Arrow support not enabled! Compile with -Darrow flag to run on SIMD backend."
{% end %}

# State-Space form derived from Newton's 2nd Law:
a_plant = [[0.0,  1.0],
           [-2.0, -0.5]].to_tensor
b_plant = [[0.0], [1.0]].to_tensor
c_plant = [[1.0,  0.0]].to_tensor
d_plant = [[0.0]].to_tensor

plant = CrySpace::StateSpace.new(a_plant, b_plant, c_plant, d_plant)
puts "▶ SS plant created: #{plant.n_states} states"

# LQR State-Feedback Design
q_lqr = [[100.0, 0.0], [0.0, 1.0]].to_tensor
r_lqr = [[0.5]].to_tensor
k_lqr, _, _ = plant.lqr(q_lqr, r_lqr)
l_lbg = plant.acker_obs([-15.0, -20.0])

# Time vector (we use a larger range or higher resolution to show off SIMD performance)
t_sim = Float64Tensor.linear_space(0.0, 10.0, 5000)
x0_sim = [[0.5], [0.0]].to_tensor
x0_est = [[0.0], [0.0]].to_tensor

# If Arrow compiled, promote tensors
{% if flag?(:arrow) %}
  t_sim = t_sim.arrow
  x0_sim = x0_sim.arrow
  x0_est = x0_est.arrow
  k_lqr = k_lqr.arrow
  l_lbg = l_lbg.arrow
{% end %}

u_sim = Float64Tensor.zeros([1, 5000])
{% if flag?(:arrow) %}
  u_sim = u_sim.arrow
{% end %}

puts "\nRunning closed-loop simulation (LQR + Observer)..."
t_out, x_out, y_out = plant.simulate(t_sim, x0: x0_sim, u: u_sim)

puts "Simulated #{t_out.size} points on backend: #{t_out.class}"
puts "First state amplitude at t=0s: #{x_out[0, 0].value}"
puts "First state amplitude at t=5s: #{x_out[2500, 0].value}"

# Save results using Arrow Feather format (if Arrow is enabled)
{% if flag?(:arrow) %}
  puts "\nLogging simulation trajectory to Feather file..."
  schema = Arrow::Schema.new([
    Arrow::Field.new("time", Arrow::DataType.double),
    Arrow::Field.new("displacement", Arrow::DataType.double)
  ])
  
  t_arr = Arrow::DoubleArray.new(t_out.to_a)
  y_arr = Arrow::DoubleArray.new(x_out[..., 0].to_a)
  table = Arrow::Table.new(schema, [t_arr, y_arr])
  
  filename = "trajectory_log.feather"
  begin
    writer = Arrow::FeatherWriter.new(filename)
    writer.write(table)
    writer.close
    puts "Successfully logged trajectory to #{filename} (#{File.size(filename)} bytes)!"
    File.delete(filename) if File.exists?(filename)
  rescue ex
    puts "Feather logging failed: #{ex.message}"
  end
{% end %}

puts "\n================================================================================"
puts "  Storyboard Completed!"
puts "================================================================================"
