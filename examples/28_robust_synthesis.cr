# 28_robust_synthesis.cr
# This example demonstrates:
# 1. Pade Approximation (TransferFunction.pade) for approximating time delays
# 2. State-Space to Transfer Function conversion (ss2tf)
# 3. Peak Gain (H-infinity norm) calculation (peak_gain)
# 4. Loop Margins calculation via Nyquist analysis (loop_margins)
# 5. H2 (h2syn) and H-infinity (hinfsyn) state-feedback optimal controller design

require "../src/cryspace"

puts "=== 1. PADE TIME DELAY APPROXIMATION ==="
# e^(-0.5 * s) approximated with a 2nd-order Pade rational transfer function
delay = 0.5
order = 2
tf_delay = CrySpace::TransferFunction.pade(delay, order)
puts "2nd-order Pade approximation for 0.5s delay:"
puts tf_delay

puts "\n=== 2. MODEL CONVERSION: SS2TF ==="
# Define continuous double integrator: dx/dt = [0 1; 0 0]*x + [0; 1]*u, y = [1 0]*x
a = [[0.0, 1.0], [0.0, 0.0]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
sys_ss = CrySpace::StateSpace.new(a, b, c, d)
tf_from_ss = sys_ss.ss2tf
puts "Converted Double Integrator StateSpace to TransferFunction:"
puts tf_from_ss

puts "\n=== 3. PEAK GAIN AND LOOP MARGINS ==="
# SISO stable system: G(s) = 1 / (s + 2)
sys_stable = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
puts "System Peak Gain: #{sys_stable.peak_gain}"

# Calculate loop margins (gain/phase margin bounds)
gm, pm = sys_stable.loop_margins
puts "Loop Margins (Nyquist distance bounds):"
puts "  Gain Margin: [#{gm[0].round(4)}, #{gm[1]}]"
puts "  Phase Margin: [#{pm[0].round(2)} deg, #{pm[1].round(2)} deg]"

puts "\n=== 4. H2 & H-INFINITY ROBUST SYNTHESIS ==="
# Define performance/weighted outputs: z = C_z * x + D_zu * u
c_z = [[1.0]].to_tensor
d_zu = [[1.0]].to_tensor

# Optimal H2 control design
k_h2, p_h2 = sys_stable.h2syn(c_z, d_zu)
puts "Optimal H2 Feedback Gain (K_h2):"
puts k_h2

# Suboptimal H-infinity design for attenuation gamma = 1.5
k_hinf, p_hinf = sys_stable.hinfsyn(c_z, d_zu, gamma: 1.5)
puts "\nRobust H-infinity Feedback Gain (K_hinf at gamma=1.5):"
puts k_hinf
