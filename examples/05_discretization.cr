# 05_discretization.cr
# This example illustrates system discretization (sampling) in CrySpace.
# It converts a continuous-time system to discrete-time using zero-order hold approximation.

require "../src/cryspace"

puts "=== 1. CONTINUOUS-TIME SYSTEM ==="
# Define a simple first-order low-pass filter: G(s) = 1 / (s + 2)
num = [1.0].to_tensor
den = [1.0, 2.0].to_tensor
sys_continuous = CrySpace::TransferFunction.new(num, den).to_statespace

puts "Continuous system matrices:"
puts sys_continuous
puts "Is discrete? #{!sys_continuous.dt.nil?}"

puts "\n=== 2. DISCRETIZATION (SAMPLING) ==="
# Discretize using a sampling time dt = 0.1 seconds
dt = 0.1
sys_discrete = sys_continuous.sample(dt)

puts "Discrete system matrices (dt = #{dt}s):"
puts sys_discrete
puts "Is discrete? #{!sys_discrete.dt.nil?}"
puts "Sampling period (dt): #{sys_discrete.dt} seconds"

puts "\n=== 3. COMPARE POLES AND STABILITY ==="
# For continuous systems: stable if Re(pole) < 0
# For discrete systems: stable if |pole| < 1
c_poles = sys_continuous.poles
d_poles = sys_discrete.poles

puts "Continuous Pole: #{c_poles[0].real.round(4)} + #{c_poles[0].imag.round(4)}i (Stable? #{sys_continuous.is_stable?})"
puts "Discrete Pole:   #{d_poles[0].real.round(4)} + #{d_poles[0].imag.round(4)}i (Stable? #{sys_discrete.is_stable?})"
puts "Note: e^(s*dt) = e^(-2 * 0.1) = e^(-0.2) = #{Math.exp(-2 * dt).round(4)} (matches discrete pole value!)"
