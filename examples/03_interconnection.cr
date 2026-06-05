# 03_interconnection.cr
# This example demonstrates system interconnection techniques in CrySpace.
# It shows how to connect state-space systems in series (*), parallel (+),
# and closed-loop feedback configurations, both with static gain matrices and dynamic systems.

require "../src/cryspace"

# Define System 1 (e.g. RLC Plant)
# G1(s) = 20 / (s^2 + 2s + 20)
num_plant = [20.0].to_tensor
den_plant = [1.0, 2.0, 20.0].to_tensor
plant = CrySpace::TransferFunction.new(num_plant, den_plant).to_statespace

# Define System 2 (e.g. PID Controller)
# We'll use a simple Proportional-Derivative (PD) controller: C(s) = 2s + 3
# StateSpace equivalent can be converted from TF: C(s) = (2s + 3) / (0.01s + 1) [filtered]
num_ctrl = [2.0, 3.0].to_tensor
den_ctrl = [0.01, 1.0].to_tensor
controller = CrySpace::TransferFunction.new(num_ctrl, den_ctrl).to_statespace

puts "=== 1. SERIES INTERCONNECTION (Forward Path) ==="
# Forward path is plant * controller
# In state-space, this forms an augmented state-space combining both system dynamics.
sys_forward = plant * controller
puts "Forward Path System states: #{sys_forward.n_states}"
puts "Forward Path matrix A dimensions: #{sys_forward.a.shape}"

puts "\n=== 2. PARALLEL INTERCONNECTION ==="
# Parallel sum of two systems
sys_parallel = plant + plant
puts "Parallel Path System states: #{sys_parallel.n_states}"

puts "\n=== 3. CLOSED-LOOP FEEDBACK CONFIGURATIONS ==="

# Configuration A: Unity Negative Feedback (Static feedback gain = 1.0)
# Closed loop: G_cl = G_f / (1 + G_f * H) with H = 1.0
# The feedback gain must be a 1x1 matrix tensor matching the dimensions of SISO inputs/outputs.
unity_gain = [[1.0]].to_tensor
sys_cl_unity = sys_forward.feedback(unity_gain)
puts "Unity Feedback system poles: #{sys_cl_unity.poles.map { |p| "#{p.real.round(4)} + #{p.imag.round(4)}i" }}"

# Configuration B: Feedback with Dynamic Lag (e.g., Sensor dynamics)
# Let's define sensor dynamics: H(s) = 1 / (0.05s + 1)
num_sensor = [1.0].to_tensor
den_sensor = [0.05, 1.0].to_tensor
sensor = CrySpace::TransferFunction.new(num_sensor, den_sensor).to_statespace

# Under negative feedback with sensor dynamics:
sys_cl_sensor = sys_forward.feedback(sensor)
puts "Sensor-feedback system states: #{sys_cl_sensor.n_states}"
puts "Sensor-feedback system poles: #{sys_cl_sensor.poles.map { |p| "#{p.real.round(4)} + #{p.imag.round(4)}i" }}"
puts "Is the sensor-feedback system stable? #{sys_cl_sensor.is_stable?}"
