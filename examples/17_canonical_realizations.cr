# 17_canonical_realizations.cr
# This example covers system canonical realizations: Observability Canonical Form 
# and Modal (Diagonalized/Decoupled) Form, converting systems to different representations.

require "../src/cryspace"

puts "=== 1. DEFINE ORIGINAL SISO SYSTEM ==="
# Transfer Function: G(s) = 2 / (s^2 + 3s + 2)
num = [2.0].to_tensor
den = [1.0, 3.0, 2.0].to_tensor
sys_c = CrySpace::TransferFunction.new(num, den).to_statespace # Controllability canonical form
puts "Controllability canonical matrices (A_c, B_c, C_c):"
puts sys_c

puts "\n=== 2. OBSERVABILITY CANONICAL FORM ==="
# Reconstruct in Observability canonical form (A_o, B_o, C_o)
sys_o = sys_c.to_observability_form
puts sys_o
# Verify that they have the same transfer function / poles:
puts "Observability poles: #{sys_o.poles.map { |p| "#{p.real.round(2)} + #{p.imag.round(2)}i" }}"

puts "\n=== 3. MODAL (DIAGONALIZED) CANONICAL FORM ==="
# Converts the state-space matrices such that A becomes diagonal/block-diagonal
sys_m = sys_c.to_modal_form
puts sys_m
# Verify that the modal A matrix is diagonal (poles are at -1 and -2)
puts "Modal poles: #{sys_m.poles.map { |p| "#{p.real.round(2)} + #{p.imag.round(2)}i" }}"
