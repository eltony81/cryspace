require "../src/cryspace"

puts "=================================================="
puts "CrySpace: Showcasing New Advanced Features"
puts "=================================================="

# 1. Runge-Kutta-Fehlberg 4(5) Adaptive ODE Solver
puts "\n1. RK45 Adaptive ODE Solver:"
# dx/dt = -2x
f = ->(x : Float64Tensor, t : Float64) { x * -2.0 }
x0 = [1.0].to_tensor
times, states = CrySpace::Solver.rk45(f, x0, {0.0, 2.0})
puts "Simulated state at t=#{times.last.round(4)}s: #{states.last[0].value.round(6)}"
puts "Exact theoretical state at t=2.0s: #{(Math.exp(-4.0)).round(6)}"
puts "Number of adaptive steps taken: #{times.size}"

# 2. Jordan Canonical Form Realization
puts "\n2. Jordan Canonical Form Realization:"
a = [[-1.0, 0.0], [0.0, -2.0]].to_tensor
b = [[1.0], [1.0]].to_tensor
c = [[1.0, 1.0]].to_tensor
d = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a, b, c, d)
sys_j, t_mat = sys.to_jordan_form
puts "Jordan Form A Matrix (modal form):"
puts sys_j.a
puts "Transformation matrix T:"
puts t_mat

# 3. Luenberger Observer State Estimation
puts "\n3. Luenberger Observer (Continuous & Discrete):"
# Plant system G(s) = 1 / (s + 2)
plant = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
# Set observer poles to -6 (so L = 4.0)
l = [[4.0]].to_tensor
observer = CrySpace::LuenbergerObserver.new(plant, l, x0_hat: [[0.0]].to_tensor)

puts "Initial estimated state: #{observer.x_hat[0, 0].value}"
# Simulate one continuous time update step with input u=1.0 and measurement y=0.5
observer.update_continuous([[0.5]].to_tensor, [[1.0]].to_tensor, 0.1)
puts "Estimated state after 0.1s continuous RK4 integration: #{observer.x_hat[0, 0].value.round(6)}"

# 4. Interactive Step & Bode Plot Dashboards
puts "\n4. Generating Plotting Dashboards:"
step_filename = "step_response.html"
bode_filename = "bode_response.html"

sys.step_plot(step_filename)
sys.bode_plot(bode_filename)
puts "Interactive Step Response Dashboard generated: #{step_filename}"
puts "Interactive Bode Plot Dashboard generated: #{bode_filename}"
puts "=================================================="
