# 10_stability_margins.cr
# This example demonstrates calculating Bode stability margins (Gain Margin and Phase Margin)
# for a SISO system. It shows how to assess stability margins of the open loop to ensure 
# closed-loop robustness.

require "../src/cryspace"

puts "=== DEFINE LOOP TRANSFER FUNCTION ==="
# Open-loop system: L(s) = 4 / (s^2 + 2s + 2)
# Under unity negative feedback, this is stable but has specific phase and gain margins.
num = [4.0].to_tensor
den = [1.0, 2.0, 2.0].to_tensor
sys = CrySpace::TransferFunction.new(num, den).to_statespace
puts sys

puts "\n=== COMPUTE STABILITY MARGINS ==="
gm, gm_db, pm, w_gc, w_pc = sys.stability_margins

puts "Gain Margin (absolute):   #{gm == Float64::INFINITY ? "Inf" : gm.round(4)}"
puts "Gain Margin (dB):         #{gm_db == Float64::INFINITY ? "Inf" : gm_db.round(2)} dB"
puts "Phase Margin (degrees):   #{pm.round(2)} deg"
puts "Gain Crossover Freq (w):  #{w_gc > 0.0 ? "#{w_gc.round(4)} rad/s" : "N/A"}"
puts "Phase Crossover Freq (w): #{w_pc > 0.0 ? "#{w_pc.round(4)} rad/s" : "N/A"}"

puts "\nNote: Since the phase never drops to -180 degrees, the Gain Margin is infinite (system is unconditionally stable for any positive gain increase)."
