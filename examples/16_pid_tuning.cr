# 16_pid_tuning.cr
# This example covers PID controller tuning using Ziegler-Nichols open-loop tuning rule
# on a First-Order Plus Dead-Time (FOPDT) system.

require "../src/cryspace"

puts "=== 1. SYSTEM CHARACTERISTICS ==="
# Suppose we have a slow process with a delay (dead time):
# Process Gain (K) = 1.5
# Time Constant (tau) = 3.0 seconds
# Dead Time (theta) = 0.2 seconds
k_process = 1.5
tau_process = 3.0
theta_process = 0.2

puts "Process characteristics:"
puts " - Gain (K):          #{k_process}"
puts " - Time Constant (T): #{tau_process} s"
puts " - Dead Time (theta): #{theta_process} s"

puts "\n=== 2. COMPUTE ZIEGLER-NICHOLS PID GAINS ==="
# Compute the optimal PID gains (Kp, Ki, Kd) using the Z-N open-loop rule
kp, ki, kd = CrySpace::Tuning.ziegler_nichols_fopdt(k_process, tau_process, theta_process)

puts "Tuned PID Controller Parameters:"
puts " - Proportional Gain (Kp):  #{kp.round(4)}"
puts " - Integral Gain (Ki):      #{ki.round(4)}"
puts " - Derivative Gain (Kd):    #{kd.round(4)}"

# Define derivative filter TF: C_pid(s) = Kp + Ki/s + (Kd*s)/(tf*s + 1)
# With a time constant tf = 0.01 seconds.
tf = 0.01
num_pid = [(kp*tf + kd), (kp + ki*tf), ki].to_tensor
den_pid = [tf, 1.0, 0.0].to_tensor
pid_controller = CrySpace::TransferFunction.new(num_pid, den_pid)

puts "\nPID Controller Transfer Function:"
puts pid_controller
