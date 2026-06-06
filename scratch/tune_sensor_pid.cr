require "../src/cryspace"

# Plant (RLC)
R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)
tf = 0.01

sensor_tau = 0.05
sensor_num = [1.0].to_tensor
sensor_den = [sensor_tau, 1.0].to_tensor
sensor_dynamics = CrySpace::TransferFunction.new(sensor_num, sensor_den).to_statespace

def simulate_sensor_pid(rlc_plant, sensor_dynamics, kp, ki, kd, tf)
  pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
  pid_den = [tf, 1.0, 0.0].to_tensor
  pid_controller = CrySpace::TransferFunction.new(pid_num, pid_den).to_statespace
  sys_forward = rlc_plant * pid_controller
  sys_cl = sys_forward.feedback(sensor_dynamics)
  
  t_vec = Float64Tensor.linear_space(0.0, 15.0, 1501)
  u_step = Float64Tensor.ones([1, 1501])
  _, _, y_cl = sys_cl.simulate(t_vec, u: u_step.dup)
  
  max_val = y_cl.max
  overshoot = max_val - 1.0
  overshoot_pct = overshoot > 0.0 ? overshoot * 100.0 : 0.0
  
  # Find settling time (within 2% of final value 1.0)
  settling_time = 15.0
  (1500).downto(0) do |i|
    val = y_cl[i, 0].value
    if (val - 1.0).abs > 0.02
      settling_time = t_vec[i].value
      break
    end
  end
  
  {kp, ki, kd, overshoot_pct, settling_time}
end

# Current parameters
puts "Current (4.0, 8.0, 1.3): #{simulate_sensor_pid(rlc_plant, sensor_dynamics, 4.0, 8.0, 1.3, tf)}"

# Search space: we want smaller overshoot. 
# Reducing Kp/Ki and adjusting Kd usually helps with sensor delay.
best_overshoot = 100.0
best_params = {0.0, 0.0, 0.0, 100.0, 15.0}

[1.0, 1.5, 2.0, 2.5, 3.0].each do |kp_val|
  [1.0, 2.0, 3.0, 4.0, 5.0].each do |ki_val|
    [0.5, 0.8, 1.0, 1.2, 1.5, 1.8].each do |kd_val|
      kp_val, ki_val, kd_val, os, st = simulate_sensor_pid(rlc_plant, sensor_dynamics, kp_val, ki_val, kd_val, tf)
      if os < 5.0 && st < 3.0
        puts "Candidate: Kp=#{kp_val}, Ki=#{ki_val}, Kd=#{kd_val} -> OS=#{os.round(2)}%, ST=#{st.round(2)}s"
      end
    end
  end
end
