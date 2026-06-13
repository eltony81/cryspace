# examples/arrow_vs_cpu_benchmark.cr
# This benchmark compares the performance of a StateSpace simulation
# with and without the Apache Arrow backend using a closed-loop RLC PID control system.
#
# Run with: crystal run --release -Darrow examples/arrow_vs_cpu_benchmark.cr

require "../src/cryspace"
require "benchmark"

# 1. Plant (RLC)
R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)

# 2. PID Controller
kp, ki, kd = 2.2, 14.6, 1.0
tf = 0.01
pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
pid_den = [tf, 1.0, 0.0].to_tensor
pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace

# 3. Closed-Loop System (4 States)
sys_cl = (rlc_plant * pid_controller).feedback([[1.0]].to_tensor)

# Time vector: 0 to 1000 seconds with dt = 0.001 (1,000,001 points)
N = 1_000_001
t_cpu = Float64Tensor.linear_space(0.0, 1000.0, N)
u_cpu = Float64Tensor.ones([1, N])

{% if flag?(:arrow) %}
  t_arr = t_cpu.arrow
  u_arr = u_cpu.arrow

  puts "Running closed-loop RLC PID simulation benchmark (1,000,000 steps, 4 states)..."
  Benchmark.ips do |x|
    x.report("CPU Simulation Loop") do
      sys_cl.simulate(t_cpu, u: u_cpu.dup)
    end

    x.report("Arrow SIMD Simulation Loop") do
      sys_cl.simulate(t_arr, u: u_arr.dup)
    end
  end
{% else %}
  puts "Arrow support not enabled. Compile with: crystal run -Darrow examples/arrow_vs_cpu_benchmark.cr"
{% end %}
