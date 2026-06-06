require "../src/cryspace"

# 1. Plant (RLC)
R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)

# 2. PID Controller
kp, ki, kd = 1.5, 1.0, 0.5
tf = 0.01
pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
pid_den = [tf, 1.0, 0.0].to_tensor
pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace

sys_cl = rlc_plant.feedback(pid_controller)

# Let's test with different step sizes
[101, 1001, 10001].each do |steps|
  t_vec = Float64Tensor.linear_space(0.0, 10.0, steps)
  u_step = Float64Tensor.ones([1, steps])
  _, _, y_cl = sys_cl.simulate(t_vec, u: u_step.dup)
  puts "Steps: #{steps}, dt: #{10.0/(steps-1)}, Max y_cl: #{y_cl.max}, Min y_cl: #{y_cl.min}"
end
