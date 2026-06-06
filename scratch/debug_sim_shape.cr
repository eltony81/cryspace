require "../src/cryspace"

kp, ki, kd = 1.5, 1.0, 0.5
tf = 0.01

pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
pid_den = [tf, 1.0, 0.0].to_tensor

pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace

R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)

sys_cl = rlc_plant.feedback(pid_controller)

t_vec = Float64Tensor.linear_space(0.0, 10.0, 101)
u_step = Float64Tensor.ones([1, 101])

# Let's print out the shapes or debug the simulate
puts "u_step shape: #{u_step.shape}"
begin
  _, _, y_cl = sys_cl.simulate(t_vec, u: u_step.dup)
  puts "Simulation completed. y_cl shape: #{y_cl.shape}"
  puts "First few values: #{y_cl[0...5, 0]}"
rescue e : Exception
  puts "Error: #{e.message}"
  puts e.backtrace.join("\n")
end
