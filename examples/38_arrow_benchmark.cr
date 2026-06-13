require "../src/cryspace"
require "benchmark"

# This benchmark compares the theoretical/actual performance of zero-copy 
# Arrow wrapping vs a copy-based approach.
# 
# To run this benchmark, the system must have 'libarrow-glib' installed.
# Run with: crystal run -Darrow examples/38_arrow_benchmark.cr

{% if flag?(:arrow) %}
  # Simulated copy-based conversion
  def copy_to_arrow_tensor(arr : Tensor(U, ARROW(U))) : Arrow::Tensor forall U
    dtype = case U
            when Float64.class then Arrow::DataType.double
            else raise "Unsupported"
            end
    
    # 1. Allocate new memory and copy data (Simulated copying)
    size_in_bytes = arr.shape.product * sizeof(U)
    copied_bytes = Bytes.new(size_in_bytes)
    arr.to_unsafe.copy_to(copied_bytes.pointer(size_in_bytes).unsafe_as(Pointer(U)), arr.shape.product)
    
    # 2. Wrap the copied buffer
    buffer = Arrow::Buffer.new(copied_bytes)
    
    shape_i64 = arr.shape.map(&.to_i64)
    strides_i64 = arr.strides.map { |s| (s * sizeof(U)).to_i64 }
    Arrow::Tensor.new(dtype, buffer, shape_i64, strides_i64)
  end

  # Setup a large Float64 Tensor (10 Million elements ~ 80MB)
  puts "Allocating Tensor of 10M Float64 elements (~80 MB)..."
  n = 10_000_000
  t = Tensor(Float64, ARROW(Float64)).new([n])

  puts "\nRunning Benchmark..."
  Benchmark.ips do |x|
    x.report("Zero-Copy Integration") do
      t.to_arrow_tensor
    end

    x.report("Copy-Based Integration (with memcpy)") do
      copy_to_arrow_tensor(t)
    end
  end
{% else %}
  puts "Arrow support not enabled. Compile with -Darrow"
  puts "\nTheoretical Performance Comparison:"
  puts "--------------------------------------------------------"
  puts "Operation        | Time Complexity | Space Complexity"
  puts "--------------------------------------------------------"
  puts "Zero-Copy        | O(1)            | O(1)"
  puts "Copy-Based       | O(N) (linear)   | O(N) (linear)"
  puts "--------------------------------------------------------"
  puts "\nFor a Tensor with 10,000,000 Float64 elements (~80 MB):"
  puts "- Zero-Copy wrapper only passes the raw pointer to GArrowBuffer, taking sub-microsecond time regardless of size."
  puts "- Copy-Based conversion requires allocating 80 MB of fresh memory and copying 10,000,000 elements, introducing allocation overhead and CPU cache pollution."
{% end %}
