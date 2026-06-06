require "../src/cryspace"

kp, ki, kd = 1.5, 1.0, 0.5
tf = 0.01

pid_num = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
pid_den = [tf, 1.0, 0.0].to_tensor

# print TF
tf_sys = CrySpace::TransferFunction.new(pid_num, pid_den)
puts tf_sys.to_s

pid_controller = tf_sys.to_statespace
puts "A:\n#{pid_controller.a}"
puts "B:\n#{pid_controller.b}"
puts "C:\n#{pid_controller.c}"
puts "D:\n#{pid_controller.d}"

# Plant (RLC)
R = 1.0; L = 0.5; C = 0.1
a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
rlc_plant = CrySpace::StateSpace.new(a, b, c, d)
puts "RLC A:\n#{rlc_plant.a}"

# Closed loop
begin
  sys_cl = rlc_plant.feedback(pid_controller)
  puts "Closed loop poles: #{sys_cl.poles}"
rescue e : Exception
  puts "Error during feedback: #{e.message}"
end
