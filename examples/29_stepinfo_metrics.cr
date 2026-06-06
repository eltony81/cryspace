# 29_stepinfo_metrics.cr
# This example demonstrates:
# 1. Step Response Performance Metrics (stepinfo)
# 2. Lead/Lag compensator transfer function design

require "../src/cryspace"

puts "=== 1. LEAD & LAG COMPENSATOR DESIGN ==="
# Design a lead compensator with zero at 2.0 rad/s and pole at 10.0 rad/s
lead = CrySpace::TransferFunction.lead_compensator(gain: 2.5, zero: 2.0, pole: 10.0)
puts "Lead Compensator Transfer Function:"
puts lead

# Design a lag compensator with zero at 0.5 rad/s and pole at 0.05 rad/s
lag = CrySpace::TransferFunction.lag_compensator(gain: 1.0, zero: 0.5, pole: 0.05)
puts "\nLag Compensator Transfer Function:"
puts lag

puts "\n=== 2. STEP RESPONSE TRANSIENT METRICS (stepinfo) ==="
# Underdamped system: G(s) = 4 / (s^2 + 2s + 4) => w_n = 2.0, zeta = 0.5
a = [[0.0, 1.0], [-4.0, -2.0]].to_tensor
b = [[0.0], [4.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a, b, c, d)

# Compute performance metrics on step response
info = sys.stepinfo(n_steps: 400)
puts "Step response metrics for 4 / (s^2 + 2s + 4):"
puts "  Steady-State Value: #{info.steady_state_value.round(4)}"
puts "  Rise Time (10% to 90%): #{info.rise_time.round(4)}s"
puts "  Settling Time (2% band): #{info.settling_time.round(4)}s"
puts "  Peak Value: #{info.peak.round(4)}"
puts "  Peak Time: #{info.peak_time.round(4)}s"
puts "  Overshoot: #{info.overshoot.round(2)}%"
