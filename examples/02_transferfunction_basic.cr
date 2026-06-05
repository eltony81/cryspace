# 02_transferfunction_basic.cr
# This example covers the TransferFunction class in CrySpace.
# It defines transfer functions, computes poles and zeros, performs arithmetic
# (addition, multiplication, feedback), and converts back and forth to StateSpace.

require "../src/cryspace"

puts "=== 1. CREATING TRANSFER FUNCTIONS ==="
# Define a Transfer Function G(s) = 2 / (s^2 + 3s + 2)
# Numerator: [2.0]
# Denominator: [1.0, 3.0, 2.0] which represents s^2 + 3s + 2
g_num = [2.0].to_tensor
g_den = [1.0, 3.0, 2.0].to_tensor
g = CrySpace::TransferFunction.new(g_num, g_den)
puts g

puts "\n=== 2. POLES AND ZEROS ==="
# Compute poles (roots of denominator) and zeros (roots of numerator)
puts "Poles of G(s): #{g.poles.map { |p| "#{p.real.round(4)} + #{p.imag.round(4)}i" }}"
puts "Zeros of G(s): #{g.zeros.map { |z| "#{z.real.round(4)} + #{z.imag.round(4)}i" }}"

puts "\n=== 3. TRANSFER FUNCTION ARITHMETIC ==="
# Let's define another transfer function H(s) = 1 / (s + 5)
h_num = [1.0].to_tensor
h_den = [1.0, 5.0].to_tensor
h = CrySpace::TransferFunction.new(h_num, h_den)
puts "H(s): #{h}"

# Series connection: G(s) * H(s)
g_series = g * h
puts "Series G(s) * H(s): #{g_series}"

# Parallel connection: G(s) + H(s)
g_parallel = g + h
puts "Parallel G(s) + H(s): #{g_parallel}"

# Feedback connection: G(s) / (1 + G(s)*H(s))
g_feedback = g.feedback(h)
puts "Feedback G(s) with H(s) in feedback path: #{g_feedback}"

puts "\n=== 4. CONVERTING TO AND FROM STATESPACE ==="
# Convert G(s) to StateSpace representation (Controllable Canonical Form)
sys_ss = g.to_statespace
puts "State-Space conversion:"
puts sys_ss

# Convert the StateSpace representation back to a Transfer Function
g_back = sys_ss.to_transferfunction
puts "Converted back to Transfer Function:"
puts g_back
