# 20_bode_nyquist_data.cr
# This example covers generating frequency response data for Bode and Nyquist plots.
# It computes log-spaced magnitudes (dB), phases (degrees), and complex real/imaginary parts.

require "../src/cryspace"

puts "=== 1. DEFINE SYSTEM MODEL ==="
# A first-order low pass filter: G(s) = 5 / (s + 5)
num = [5.0].to_tensor
den = [1.0, 5.0].to_tensor
sys = CrySpace::TransferFunction.new(num, den).to_statespace
puts sys

puts "\n=== 2. GENERATE BODE RESPONSE DATA ==="
# Frequencies (rad/s): 0.1, 1.0, 5.0, 10.0, 100.0
omega = [0.1, 1.0, 5.0, 10.0, 100.0].to_tensor

w_out, db, deg = sys.bode_data(omega)

puts "Freq (rad/s) | Magnitude (dB) | Phase (deg)"
puts "-------------------------------------------"
omega.size.times do |i|
  puts "  #{w_out[i].value.to_s.ljust(10)} |  #{db[i].round(2).to_s.ljust(13)} |  #{deg[i].round(1)}"
end

puts "\n=== 3. GENERATE NYQUIST DATA ==="
real_parts, imag_parts = sys.nyquist_data(omega)

puts "Freq (rad/s) | Real Part | Imaginary Part"
puts "-----------------------------------------"
omega.size.times do |i|
  puts "  #{w_out[i].value.to_s.ljust(10)} |  #{real_parts[i].round(4).to_s.ljust(8)} |  #{imag_parts[i].round(4)}"
end
