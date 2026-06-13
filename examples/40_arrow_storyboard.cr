# examples/40_arrow_storyboard.cr
# Specular version of 36_mega_usecase_storyboard.cr showcasing how to integrate Apache Arrow SIMD.
#
# Compilation Instructions:
#   To compile and run this script with Apache Arrow SIMD optimizations enabled,
#   make sure you have installed 'libarrow-glib-dev' (Ubuntu/Debian) or 'arrow' (Arch/Manjaro).
#
#   Compile / Run Command:
#     crystal run -Darrow examples/40_arrow_storyboard.cr
#
#   Build Production Binary Command:
#     crystal build -Darrow --release examples/40_arrow_storyboard.cr
#

require "../src/cryspace"

puts "================================================================================"
puts "  CrySpace Complete Storyboard Tutorial (Apache Arrow SIMD Showcase)"
puts "  Case Study: Active Vehicle Suspension Control System Design"
puts "================================================================================"

{% if flag?(:arrow) %}
  puts "Arrow support enabled!"

  # ==============================================================================
  # Chapter 1 — Plant Modeling & Representation
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 1 · Plant Modeling & Representation"
  puts "═══════════════════════════════════════════════════════════════"

  # State-Space form derived from Newton's 2nd Law:
  a_plant = [[0.0,  1.0],
             [-2.0, -0.5]].to_tensor
  b_plant = [[0.0], [1.0]].to_tensor
  c_plant = [[1.0,  0.0]].to_tensor
  d_plant = [[0.0]].to_tensor

  plant = CrySpace::StateSpace.new(a_plant, b_plant, c_plant, d_plant)
  puts "▶ SS plant created:  #{plant.n_states} states, #{plant.n_inputs} input, #{plant.n_outputs} output"

  # Convert to Transfer Function G(s) = Y(s)/U(s)
  plant_tf = plant.to_transferfunction
  puts "\n▶ Transfer Function G(s):"
  puts plant_tf

  # DC gain
  dc = plant.dcgain
  puts "▶ DC gain (static compliance x/F at s=0): #{dc[0,0].value.round(4)} m/N"


  # ==============================================================================
  # Chapter 2 — Stability & Canonical Forms
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 2 · Stability Analysis & Canonical Realizations"
  puts "═══════════════════════════════════════════════════════════════"

  # Check if the plain plant is controllable and observable
  puts "▶ Plant controllability: #{plant.is_controllable?}"
  puts "▶ Plant observability:   #{plant.is_observable?}"

  # Transform to Control Canonical Form
  plant_ccf, t_ccf = plant.to_control_canonical_form
  puts "\n▶ Control Canonical Form A matrix:"
  puts plant_ccf.a

  # Transform to Jordan / Modal Canonical Form
  plant_jcf, t_jcf = plant.to_jordan_form
  puts "\n▶ Jordan Canonical Form A matrix (modal dynamics):"
  puts plant_jcf.a


  # ==============================================================================
  # Chapter 3 — Frequency Domain Analysis
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 3 · Frequency Domain Analysis"
  puts "═══════════════════════════════════════════════════════════════"

  # Bode data over relevant frequency range (CPU backend required for bode_data)
  omega_vals = [0.1, 0.3, 1.0, 3.0, 10.0, 50.0]
  omega_cpu = omega_vals.to_tensor
  w_out, mags_db, phases_deg = plant.bode_data(omega_cpu)

  puts "▶ Bode frequency response:"
  omega_cpu.size.times do |i|
    puts "  ω = #{w_out[i].value.round(2).to_s.rjust(6)} rad/s │ " \
         "#{mags_db[i].round(2).to_s.rjust(8)} dB │ #{phases_deg[i].round(1).to_s.rjust(8)} °"
  end


  # ==============================================================================
  # Chapter 4 — LQR State-Feedback Design & Tracking
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 4 · LQR State-Feedback & Prefilter Design"
  puts "═══════════════════════════════════════════════════════════════"

  # LQR
  q_lqr = [[100.0, 0.0], [0.0, 1.0]].to_tensor
  r_lqr = [[0.5]].to_tensor
  k_lqr, p_care, cl_poles_lqr = plant.lqr(q_lqr, r_lqr)
  puts "▶ LQR gain K:              #{k_lqr.to_a.flatten.map { |x| x.round(4) }.join(", ")}"

  # Reference prefilter Nbar
  n_bar = plant.nbar(k_lqr)
  puts "▶ Prefilter Nbar:          #{n_bar[0,0].value.round(4)}"


  # ==============================================================================
  # Chapter 5 — Observer Design: Kalman & Luenberger
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 5 · State Estimation: LQE / Luenberger Observer"
  puts "═══════════════════════════════════════════════════════════════"

  # LQE (Kalman Estimator)
  q_noise = [[0.5, 0.0], [0.0, 0.1]].to_tensor
  r_noise = [[0.05]].to_tensor
  l_lqe   = plant.lqe(q_noise, r_noise)
  puts "▶ LQE Kalman Observer gain L: [#{l_lqe[0,0].value.round(4)}, #{l_lqe[1,0].value.round(4)}]"

  # Luenberger Observer
  l_lbg = plant.acker_obs([-15.0, -20.0])
  puts "\n▶ Luenberger gain (poles @ -15, -20): [#{l_lbg[0,0].value.round(4)}, #{l_lbg[1,0].value.round(4)}]"

  # Luenberger continuous state estimation
  luenberger = CrySpace::LuenbergerObserver.new(plant, l_lbg, x0_hat: [[0.0], [0.0]].to_tensor)
  luenberger.update_continuous([[0.4]].to_tensor, [[0.0]].to_tensor, 0.05)
  puts "▶ Luenberger x_hat after 1 RK4 step: [#{luenberger.x_hat[0,0].value.round(5)}, #{luenberger.x_hat[1,0].value.round(5)}]"


  # ==============================================================================
  # Chapter 6 — Discretization
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 6 · Discretization Methods"
  puts "═══════════════════════════════════════════════════════════════"

  dt = 0.05
  plant_zoh     = plant.sample(dt)
  plant_tustin  = plant.sample(dt, method: :tustin)
  puts "▶ ZOH A₁₁:          #{plant_zoh.a[0,0].value.round(6)}"
  puts "▶ Tustin A₁₁:       #{plant_tustin.a[0,0].value.round(6)}"


  # ==============================================================================
  # Chapter 7 — Model Order Reduction
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 7 · Model Order Reduction"
  puts "═══════════════════════════════════════════════════════════════"

  sys_high = (plant_tf * CrySpace::TransferFunction.pade(0.1, 1)).to_statespace
  sys_bt = sys_high.balred(2)
  puts "▶ High-order plant+delay system: #{sys_high.n_states} states"
  puts "▶ After Balanced Truncation → #{sys_bt.n_states} states"


  # ==============================================================================
  # Chapter 8 — PID Design & Anti-Windup
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 8 · PID Tuning & Anti-Windup"
  puts "═══════════════════════════════════════════════════════════════"

  zn = CrySpace::Tuning.pid_tune_zn(ku: 8.0, tu: 1.2, type: :pid)
  puts "▶ ZN PID: Kp=#{zn[:kp].round(3)}, Ki=#{zn[:ki].round(3)}, Kd=#{zn[:kd].round(3)}"


  # ==============================================================================
  # Chapter 9 — Describing Functions
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 9 · Nonlinear Analysis via Describing Functions"
  puts "═══════════════════════════════════════════════════════════════"

  df = CrySpace::Nonlinear.describing_function_saturation(4.0, 3.0)
  puts "  Amplitude 4.0 N → Saturation DF gain = #{df.round(5)}"


  # ==============================================================================
  # Chapter 10 — System Identification
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 10 · System Identification"
  puts "═══════════════════════════════════════════════════════════════"

  u_id = Float64Tensor.zeros([40])
  y_id = Float64Tensor.zeros([40])
  sys_ident = CrySpace::Ident.least_squares_estimation(u_id, y_id, 1, 0.1)
  puts "▶ System identification completed."


  # ==============================================================================
  # Chapter 11 — Nonlinear ODE Solvers
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 11 · Nonlinear ODE Solvers"
  puts "═══════════════════════════════════════════════════════════════"

  f_nl = ->(x : Float64Tensor, t : Float64) {
    x1 = x[0].value
    x2 = x[1].value
    dx1 = x2
    dx2 = -(2.0 + 0.3*x1*x1)*x1 - 0.5*x2
    [dx1, dx2].to_tensor
  }
  x0_nl = [0.5, 0.0].to_tensor
  _, states_rk4 = CrySpace::Solver.rk4(f_nl, x0_nl, {0.0, 1.0}, 0.05)
  puts "▶ RK4 nonlinear sim: x₁(1.0s) = #{states_rk4.last[0].value.round(5)} m"


  # ==============================================================================
  # Chapter 12 — Closed-Loop Simulation with Apache Arrow SIMD Backend
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 12 · Closed-Loop Simulation (Apache Arrow SIMD)"
  puts "═══════════════════════════════════════════════════════════════"

  # Create closed-loop system: plant with LQR state feedback
  # u = -K*x + N*r -> We augment the system or simulate it directly.
  # Let's simulate the closed loop plant: A_cl = A - B*K
  a_cl = a_plant - b_plant.matmul(k_lqr)
  sys_cl = CrySpace::StateSpace.new(a_cl, b_plant * n_bar, c_plant, d_plant)

  # Setup simulation time and reference vectors
  N_steps = 1001
  t_cpu = Float64Tensor.linear_space(0.0, 10.0, N_steps)
  u_cpu = Float64Tensor.ones([1, N_steps])

  # PROMOTE to Apache Arrow backend
  t_arr = t_cpu.arrow
  u_arr = u_cpu.arrow
  x0_arr = [[0.2], [0.0]].to_tensor.arrow

  # Offloaded automatically to Arrow SIMD compute kernel due to t_arr/u_arr being Arrow-backed
  t_out, x_out, y_out = sys_cl.simulate(t_arr, x0: x0_arr, u: u_arr)

  step_times = [0, 200, 400, 600, 800, 1000]
  puts "▶ Closed-loop step response under LQR control (Arrow SIMD Offloaded):"
  puts "  #{"time".rjust(6)} │ #{"position (true)".rjust(16)} │ #{"velocity".rjust(10)}"
  step_times.each do |i|
    pos = x_out[i, 0].value
    vel = x_out[i, 1].value
    puts "  #{t_out[i].value.round(3).to_s.rjust(6)} │ #{pos.round(5).to_s.rjust(16)} │ #{vel.round(5).to_s.rjust(10)}"
  end


  # ==============================================================================
  # Chapter 13 — Arrow Feather I/O Serialization
  # ==============================================================================
  puts "\n═══════════════════════════════════════════════════════════════"
  puts " Chapter 13 · Arrow Feather Log I/O Serialization"
  puts "═══════════════════════════════════════════════════════════════"

  # Save trajectory log using Arrow Feather format
  schema = Arrow::Schema.new([
    Arrow::Field.new("time", Arrow::DataType.double),
    Arrow::Field.new("displacement", Arrow::DataType.double)
  ])
  time_array = Arrow::DoubleArray.new(t_out[..., 0].to_a)
  disp_array = Arrow::DoubleArray.new(x_out[..., 0].to_a)
  table = Arrow::Table.new(schema, [time_array, disp_array])

  filename = "suspension_trajectory.feather"
  writer = Arrow::FeatherWriter.new(filename)
  writer.write(table)
  writer.close
  puts "▶ Successfully logged suspension trajectory to memory-mapped Feather file: #{filename}"
  File.delete(filename) if File.exists?(filename)

  puts "\n================================================================================"
  puts "  Arrow Storyboard Tutorial Complete!"
  puts "================================================================================"

{% else %}
  puts "Arrow support not enabled. Compile with -Darrow"
{% end %}
