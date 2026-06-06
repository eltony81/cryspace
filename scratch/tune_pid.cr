require "../src/cryspace"

# Plant (RLC)
R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)
tf = 0.01

def simulate_pid(rlc_plant, kp, ki, kd, tf)
  pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
  pid_den = [tf, 1.0, 0.0].to_tensor
  pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace
  sys_cl = (rlc_plant * pid_controller).feedback([[1.0]].to_tensor)
  t_vec = Float64Tensor.linear_space(0.0, 15.0, 1501)
  u_step = Float64Tensor.ones([1, 1501])
  _, _, y_cl = sys_cl.simulate(t_vec, u: u_step.dup)
  
  max_val = y_cl.max
  overshoot = max_val - 1.0
  overshoot_pct = overshoot > 0.0 ? overshoot * 100.0 : 0.0
  {kp, ki, kd, overshoot_pct, y_cl[-1, 0].value}
end

# Current
puts "Current: #{simulate_pid(rlc_plant, 1.5, 1.0, 0.5, tf)}"
# Reduce Ki, Increase Kd
puts "Tune 1:  #{simulate_pid(rlc_plant, 1.5, 0.5, 0.6, tf)}"
puts "Tune 2:  #{simulate_pid(rlc_plant, 1.2, 0.3, 0.6, tf)}"
puts "Tune 3:  #{simulate_pid(rlc_plant, 1.0, 0.2, 0.5, tf)}"
puts "Tune 4:  #{simulate_pid(rlc_plant, 1.0, 0.5, 0.8, tf)}"
