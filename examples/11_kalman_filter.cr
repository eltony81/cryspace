# 11_kalman_filter.cr
# This example demonstrates a Discrete-Time Kalman Filter.
# It simulates a simple moving object (1D position & velocity) tracked by a noisy sensor,
# showing how the Kalman filter fuses predictions and measurements to reconstruct the true state.

require "../src/cryspace"

puts "=== 1. DEFINE SYSTEM DYNAMICS ==="
# A first-order kinematic model (constant velocity model with sampling period dt = 0.5s)
# States: x1 = position, x2 = velocity
# x_k+1 = A * x_k + B * u_k
dt = 0.5
a = [[1.0, dt], [0.0, 1.0]].to_tensor
b = [[0.5 * dt**2], [dt]].to_tensor # acceleration input
c = [[1.0, 0.0]].to_tensor          # we only measure position directly
d = [[0.0]].to_tensor
sys_discrete = CrySpace::StateSpace.new(a, b, c, d, dt)
puts sys_discrete

# 2. Kalman Covariances
# Process noise covariance (model uncertainty)
q = [[0.05, 0.0], [0.0, 0.01]].to_tensor
# Measurement noise covariance (sensor uncertainty)
r = [[1.5]].to_tensor

# 3. Initialize Kalman Filter
# Initial state estimate [0, 0]^T and initial high covariance
x0 = [[0.0], [0.0]].to_tensor
p0 = [[10.0, 0.0], [0.0, 10.0]].to_tensor
kf = CrySpace::KalmanFilter.new(sys_discrete, q, r, x0, p0)

puts "\n=== 2. RUN KALMAN FILTER TRACE ==="
# True initial state
x_true = [[0.0], [2.0]].to_tensor # starts at position 0, velocity 2 m/s
u = [[0.2]].to_tensor             # constant acceleration input

puts "Step | True Pos | True Vel | Measured Pos | Estimated Pos | Estimated Vel"
puts "--------------------------------------------------------------------------"

10.times do |step|
  # 1. Update true system state (with zero process noise for this simulation demo)
  x_true = sys_discrete.a.matmul(x_true) + sys_discrete.b.matmul(u)
  
  # 2. Generate noisy measurement: y = C * x + noise
  # We simulate noise manually (e.g. +0.5 offset deviation in this deterministic run for clarity)
  noise = [[step % 2 == 0 ? 0.8 : -0.8]].to_tensor
  y_measured = sys_discrete.c.matmul(x_true) + noise
  
  # 3. Kalman Filter: Predict step
  kf.predict(u)
  
  # 4. Kalman Filter: Update step with measurement
  kf.update(y_measured)
  
  # Print values
  puts "  #{step.to_s.ljust(2)} |  #{x_true[0, 0].value.round(2).to_s.ljust(7)} |  #{x_true[1, 0].value.round(2).to_s.ljust(7)} |  #{y_measured[0, 0].value.round(2).to_s.ljust(11)} |  #{kf.x[0, 0].value.round(2).to_s.ljust(12)} |  #{kf.x[1, 0].value.round(2)}"
end
