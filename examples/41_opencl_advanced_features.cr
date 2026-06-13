require "../src/cryspace"

{% if flag?(:opencl) %}
  puts "OpenCL GPU Acceleration support enabled!"

  # 1. Simulate a system on the GPU
  puts "=== 1. System Simulation with OpenCL Backend ==="
  num = [10.0].to_tensor
  den = [1.0, 2.0, 10.0].to_tensor
  sys = CrySpace::TransferFunction.new(num, den).to_statespace
  
  # Perform simulation
  t, x, y = sys.step_response(n_steps: 100)
  puts "Simulated #{y.size} steps successfully."

  # 2. Re-allocate matrices on OpenCL Backend for zero-copy slicing / sub-buffers
  puts "\n=== 2. Allocation & Sub-buffer zero-copy slicing ==="
  # Let's allocate a larger control system matrix (e.g. 128x128)
  # Slicing via OpenCL sub-buffers allows zero-copy block extraction.
  begin
    a_gpu = Tensor.new([128, 128], device: OCL) { |i, j| (i + j).to_f64 }
    
    # Extract the top-left 64x64 sub-system matrix using our new sub_tensor API
    # 0 offset elements, shape [64, 64], strides [128, 1]
    a_sub = a_gpu.sub_tensor(0, [64, 64], [128, 1])
    puts "Successfully created sub-system matrix via zero-copy OpenCL Sub-Buffer!"
    puts "Sub-matrix shape: #{a_sub.shape}"
  rescue ex
    puts "GPU sub-buffer check skipped or platform not available: #{ex.message}"
  end

  # 3. Dynamic Backend Dispatch
  puts "\n=== 3. Dynamic Backend Dispatch Demonstration ==="
  # Operations on tensors with >= 1,000,000 elements automatically route to GPU
  # when compiled with -Dopencl.
  large_tensor_1 = Tensor.new([1000, 1000]) { |i| 1.0 }
  large_tensor_2 = Tensor.new([1000, 1000]) { |i| 2.0 }
  
  # This dynamic addition routes to GPU
  result = large_tensor_1 + large_tensor_2
  puts "Large matrix sum result (first 5 elements): #{result.to_a[0..4]}"

{% else %}
  puts "OpenCL support not enabled. Compile with -Dopencl"
{% end %}
