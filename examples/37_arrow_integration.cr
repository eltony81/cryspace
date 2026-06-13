require "../src/cryspace"

{% if flag?(:arrow) %}
  puts "Arrow support enabled!"

  # 1. Define a system and simulate a step response
  puts "=== 1. System Simulation ==="
  num = [5.0].to_tensor
  den = [1.0, 1.0, 5.0].to_tensor
  sys = CrySpace::TransferFunction.new(num, den).to_statespace
  t_step, x_step, y_step = sys.step_response(n_steps: 10)

  # Collect outputs as flat Float64 array
  y_flat = [] of Float64
  y_step.each do |val|
    y_flat << val.value
  end
  puts "Generated #{y_flat.size} step response simulation steps."

  # 2. Place the simulation results on ARROW backend
  puts "\n=== 2. Creating Tensors on ARROW Backend ==="
  t_arrow = Tensor(Float64, ARROW(Float64)).new([y_flat.size], device: ARROW(Float64)) { |i| y_flat[i] }
  puts "Tensor on ARROW backend:"
  puts t_arrow

  # 3. Vectorized math via Arrow Compute Engine
  puts "\n=== 3. Vectorized Math (Arrow Compute Engine) ==="
  # Scales the response vector by 2.0 (delegated to C++ Arrow compute kernel)
  scaled_arrow = t_arrow + t_arrow
  puts "Scaled Tensor (2 * trajectory):"
  puts scaled_arrow

  # 4. Save simulation results to a Feather file (ultra-fast columnar binary format)
  puts "\n=== 4. Saving Simulation Trajectory to Feather file ==="
  schema = Arrow::Schema.new([
    Arrow::Field.new("time", Arrow::DataType.double),
    Arrow::Field.new("amplitude", Arrow::DataType.double)
  ])

  t_arr = Arrow::DoubleArray.new(t_step)
  y_arr = Arrow::DoubleArray.new(y_flat)
  table = Arrow::Table.new(schema, [t_arr, y_arr])

  filename = "./trajectory_log.feather"
  begin
    writer = Arrow::FeatherWriter.new(filename)
    writer.write(table)
    writer.close
    puts "Successfully logged trajectory to #{filename}!"
    if File.exists?(filename)
      puts "Feather file size: #{File.size(filename)} bytes"
      File.delete(filename)
    end
  rescue ex
    puts "Feather file writer skipped/failed: #{ex.message}"
  end

{% else %}
  puts "Arrow support not enabled. Compile with -Darrow"
{% end %}
