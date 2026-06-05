# 14_model_reduction.cr
# This example covers system Gramians (controllability and observability) and 
# Hankel Singular Values (HSVD) used for model order reduction.

require "../src/cryspace"

puts "=== 1. DEFINE SYSTEM MODEL ==="
# A stable first-order low-pass filter: G(s) = 3 / (s + 2)
# StateSpace: A = [-2], B = [1], C = [3], D = [0]
sys = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[3.0]].to_tensor, [[0.0]].to_tensor)
puts sys

puts "\n=== 2. COMPUTE CONTROLLABILITY & OBSERVABILITY GRAMIANS ==="
# Controllability Gramian (Wc) solves: A*Wc + Wc*A^T + B*B^T = 0
wc = sys.gram(:c)
puts "Controllability Gramian Wc:"
puts wc

# Observability Gramian (Wo) solves: A^T*Wo + Wo*A + C^T*C = 0
wo = sys.gram(:o)
puts "Observability Gramian Wo:"
puts wo

puts "\n=== 3. HANKEL SINGULAR VALUES ==="
# Hankel Singular Values represent the state energy contributions.
# They are the square roots of the eigenvalues of the product (Wc * Wo).
hsv = sys.hsvd
puts "Hankel Singular Values: #{hsv.map { |v| v.round(4) }}"
puts "Note: The singular value is exactly sqrt(0.25 * 2.25) = sqrt(0.5625) = 0.75"
