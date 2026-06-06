# 27_tf_algebra.cr
# This example demonstrates:
# 1. TransferFunction series (*) and parallel (+) interconnections
# 2. TransferFunction feedback loops with dynamic signs (feedback)
# 3. TransferFunction minimal realization / pole-zero cancellation (minreal)

require "../src/cryspace"

puts "=== 1. DEFINE TRANSFER FUNCTIONS ==="
# G1(s) = 1 / (s + 2)
num1 = [1.0].to_tensor
den1 = [1.0, 2.0].to_tensor
g1 = CrySpace::TransferFunction.new(num1, den1)
puts "G1(s):"
puts g1

# G2(s) = (s + 1) / (s + 3)
num2 = [1.0, 1.0].to_tensor
den2 = [1.0, 3.0].to_tensor
g2 = CrySpace::TransferFunction.new(num2, den2)
puts "\nG2(s):"
puts g2

puts "\n=== 2. TF INTERCONNECTIONS ==="
# Series: G_series = G1 * G2
g_series = g1 * g2
puts "Series G1(s) * G2(s):"
puts g_series

# Parallel: G_parallel = G1 + G2
g_parallel = g1 + g2
puts "\nParallel G1(s) + G2(s):"
puts g_parallel

puts "\n=== 3. FEEDBACK LOOP CONNECTIONS ==="
# Negative feedback: G_cl = G1 / (1 + G1) (feedback with default sign = -1)
g_cl_neg = g1.feedback(CrySpace::TransferFunction.new([1.0].to_tensor, [1.0].to_tensor))
puts "Negative Feedback G1 / (1 + G1):"
puts g_cl_neg

# Positive feedback: G_cl = G1 / (1 - G1) (feedback with sign = 1)
g_cl_pos = g1.feedback(CrySpace::TransferFunction.new([1.0].to_tensor, [1.0].to_tensor), sign: 1)
puts "\nPositive Feedback G1 / (1 - G1):"
puts g_cl_pos

puts "\n=== 4. POLE-ZERO CANCELLATION (MINREAL) ==="
# G3(s) = (s + 1) / ( (s + 1)(s + 2) ) = (s + 1) / (s^2 + 3s + 2)
# Under minreal, the (s + 1) term should cancel out, yielding 1 / (s + 2)
num3 = [1.0, 1.0].to_tensor
den3 = [1.0, 3.0, 2.0].to_tensor
g3 = CrySpace::TransferFunction.new(num3, den3)
puts "G3 before minreal:"
puts g3

g3_min = g3.minreal
puts "\nG3 after minreal (cancelled pole-zero at -1):"
puts g3_min
