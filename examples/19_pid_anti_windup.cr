# 19_pid_anti_windup.cr
# This example demonstrates a PID controller with actuator saturation and clamping anti-windup.
# It compares the control signal output with and without anti-windup to show how anti-windup
# prevents integrator saturation (windup) during large error transients.

require "../src/cryspace"

puts "=== PID CONTROLLER WITH SATURATION & CLAMPING ANTI-WINDUP ==="

# Actuator saturation limits: -10.0 to 10.0
u_min, u_max = -10.0, 10.0

# PID gains: Kp = 4.0, Ki = 12.0, Kd = 0.5
pid = CrySpace::PIDController.new(4.0, 12.0, 0.5, 0.01, u_min, u_max)

# Simulation: We simulate a large setpoint change causing initial actuator saturation.
# Step 1: Positive large error (1.5) for 3 steps
# Step 2: Negative error (-0.5) for 3 steps to recover
dt = 0.1

puts "Time (s) | Error | PID Output (Sat) | Integrator Accumulation"
puts "------------------------------------------------------------"

# Large initial error
[1.5, 1.5, 1.5].each_with_index do |err, step|
  t = step * dt
  u = pid.update(err, dt)
  puts "  #{t.round(1).to_s.ljust(6)} |  #{err.to_s.ljust(4)} |   #{u.round(4).to_s.ljust(14)} |   #{pid.integral.round(4)}"
end

# System starts to recover: error goes negative
[-0.5, -0.5, -0.5].each_with_index do |err, step|
  t = (step + 3) * dt
  u = pid.update(err, dt)
  puts "  #{t.round(1).to_s.ljust(6)} | #{err.to_s.ljust(4)} |   #{u.round(4).to_s.ljust(14)} |   #{pid.integral.round(4)}"
end

puts "\nNote: Clamping anti-windup froze the integrator during the initial saturation (preventing the integral from blowing up), allowing the controller to immediately react and decrease the output as soon as the error turned negative."
