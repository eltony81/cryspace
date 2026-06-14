# examples/arrow_vs_cpu_benchmark.cr
# This benchmark simulates a continuous closed-loop RLC PID control system
# with 4 states (A in R^4x4) using Runge-Kutta 4th-order (RK4) integration
# for 1,000,000 simulation steps (dt = 0.001s).
# It compares the CPU backend with the Apache Arrow SIMD backend (-Darrow).

require "../src/cryspace"
require "benchmark"

# 1. Setup RLC Plant + PID Controller (total 4 states)
R = 1.0; L = 0.5; C = 0.1
a_plant = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b_plant = [[0.0], [1/L]].to_tensor
c_plant = [[1.0, 0.0]].to_tensor
d_plant = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a_plant, b_plant, c_plant, d_plant)

kp, ki, kd = 2.2, 14.6, 1.0
tf = 0.01
pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
pid_den = [tf, 1.0, 0.0].to_tensor
pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace

# Closed-loop connection (4 states total)
sys_cl = (rlc_plant * pid_controller).feedback([[1.0]].to_tensor)

puts "System closed-loop states: #{sys_cl.n_states}"
puts "Generating 1,000,000 simulation steps time & input vectors..."

# Time vector (1M steps, dt = 0.001)
n_steps = 1_000_000
t_cpu = Float64Tensor.linear_space(0.0, 1000.0, n_steps)
u_cpu = Float64Tensor.ones([1, n_steps])

# Run CPU benchmark
GC.collect
m_start_cpu = GC.stats.total_bytes
time_cpu = Time.measure do
  sys_cl.simulate(t_cpu, u: u_cpu, method: :rk4)
end
m_end_cpu = GC.stats.total_bytes
allocated_cpu = (m_end_cpu - m_start_cpu) / 1024.0 / 1024.0

puts "\n=== CPU Backend Results ==="
puts "Execution Time: #{time_cpu.total_seconds.round(3)} s"
puts "Allocated Memory (approx): #{allocated_cpu.round(2)} MB"

{% if flag?(:arrow) %}
  puts "\nPromoting vectors to Arrow backend..."
  t_arrow = t_cpu.arrow
  u_arrow = u_cpu.arrow

  # Run Arrow SIMD benchmark
  GC.collect
  m_start_arrow = GC.stats.total_bytes
  time_arrow = Time.measure do
    sys_cl.simulate(t_arrow, u: u_arrow, method: :rk4)
  end
  m_end_arrow = GC.stats.total_bytes
  allocated_arrow = (m_end_arrow - m_start_arrow) / 1024.0 / 1024.0

  puts "\n=== Apache Arrow SIMD Backend Results ==="
  puts "Execution Time: #{time_arrow.total_seconds.round(3)} s"
  puts "Allocated Memory (approx): #{allocated_arrow.round(2)} MB"
  
  speedup = time_cpu.total_seconds / time_arrow.total_seconds
  puts "\nSpeedup: #{speedup.round(2)}x faster"
{% else %}
  puts "\nArrow support not enabled. Compile with -Darrow to see Arrow SIMD results."
{% end %}
