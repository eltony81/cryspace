# examples/39_arrow_math_benchmark.cr
# This benchmark compares CPU element-wise operations vs. Apache Arrow Compute Engine
# vector/SIMD operations for in-place math and negation.
#
# To run this benchmark:
#   crystal run --release -Darrow examples/39_arrow_math_benchmark.cr

require "../src/cryspace"
require "benchmark"

{% if flag?(:arrow) %}
  N = 10_000_000 # 10 Million Float64 elements (~80 MB)
  
  puts "Allocating tensors (10M Float64 elements each)..."
  cpu_a = Tensor(Float64, CPU(Float64)).new([N], device: CPU(Float64)) { |i| i.to_f }
  cpu_b = Tensor(Float64, CPU(Float64)).new([N], device: CPU(Float64)) { |i| 5.0 }
  
  arr_a = Tensor(Float64, ARROW(Float64)).new([N], device: ARROW(Float64)) { |i| i.to_f }
  arr_b = Tensor(Float64, ARROW(Float64)).new([N], device: ARROW(Float64)) { |i| 5.0 }

  puts "\n1. Benchmarking In-place Addition (add!)..."
  Benchmark.ips do |x|
    x.report("CPU In-place Add") do
      Num.add!(cpu_a, cpu_b)
    end

    x.report("Arrow Compute In-place Add (SIMD)") do
      Num.add!(arr_a, arr_b)
    end
  end

  puts "\n2. Benchmarking Unary Negation (-a)..."
  Benchmark.ips do |x|
    x.report("CPU Negation") do
      -cpu_a
    end

    x.report("Arrow Compute Negation (SIMD)") do
      -arr_a
    end
  end
{% else %}
  puts "Arrow support not enabled. Compile with: crystal run -Darrow examples/39_arrow_math_benchmark.cr"
  puts "\nTheoretical SIMD vs Element-wise loop analysis:"
  puts "-----------------------------------------------------------------------"
  puts "Operation   | CPU Loop Performance | Arrow Compute SIMD (C++)"
  puts "-----------------------------------------------------------------------"
  puts "In-place +  | Element-wise loop    | Vectorized SIMD (avx/avx512/neon)"
  puts "Negation    | Copies via map loop  | Fast C++ kernel allocation + SIMD"
  puts "-----------------------------------------------------------------------"
{% end %}
