# How to run this benchmark:
#
# 1. Ensure you have the required OpenCL drivers/runtimes and CLBlast installed.
#    On Debian/Ubuntu:
#      sudo apt-get install ocl-icd-opencl-dev clblast libclblast-dev
#    On Arch Linux:
#      sudo pacman -S opencl-headers ocl-icd clblast
#
# 2. Compile and run with the -Dopencl flag in release mode for accurate metrics:
#    crystal run -Dopencl --release examples/42_opencl_benchmark.cr
#
# 3. To run CPU-only benchmarking:
#    crystal run --release examples/42_opencl_benchmark.cr

require "../src/cryspace"
require "benchmark"

puts "=== OpenCL Performance & Slicing Benchmark ==="

# We want to measure the performance improvement of using the new zero-copy Sub-Buffers (sub_tensor)
# vs copy-based slicing on the GPU, and compare it with the CPU.

size = 512
shape = [size, size]
strides = [size, 1]

# CPU Slicing
cpu_matrix = Tensor.new(shape) { |i| i.to_f64 }.reshape(shape)

# Check OpenCL support
{% if flag?(:opencl) %}
  puts "OpenCL flag is enabled. Attempting to initialize GPU buffers..."
  
  begin
    # Force disable SVM to test clCreateSubBuffer (non-SVM backend)
    Num::ClContext.instance.svm_supported = false

    storage = OCL(Float64).new(shape, strides, 1.0)
    gpu_matrix = Tensor(Float64, OCL(Float64)).from_storage(storage, shape.map(&.to_i32), strides.map(&.to_i32), 0)
    
    puts "GPU Buffers successfully initialized. Starting benchmarks..."

    Benchmark.ips do |x|
      x.report("CPU Slice (strided view)") do
        cpu_matrix[0...256, 0...256]
      end

      x.report("GPU Slice (traditional)") do
        gpu_matrix[0...256, 0...256]
      end

      x.report("GPU Sub-Buffer (sub_tensor)") do
        # Extracts the 256x256 block starting at 0 using the new sub-buffer API
        # Offset = 0, shape = [256, 256], strides = [512, 1]
        gpu_matrix.sub_tensor(0, [256, 256], [512, 1])
      end
    end
  rescue ex : Exception
    puts "\n[Warning] Could not initialize OpenCL execution: #{ex.message}"
    puts "This typically happens if no OpenCL platforms/devices are available (e.g., in a headless container or sandbox)."
    puts "Falling back to CPU-only benchmarks.\n"
    
    run_cpu_benchmark(cpu_matrix)
  end

{% else %}
  puts "OpenCL support is not enabled (compile with -Dopencl to test GPU performance)."
  run_cpu_benchmark(cpu_matrix)
{% end %}

def run_cpu_benchmark(cpu_matrix)
  puts "Running CPU-only slicing benchmark:"
  Benchmark.ips do |x|
    x.report("CPU Slice (strided view)") do
      cpu_matrix[0...256, 0...256]
    end
  end
end

