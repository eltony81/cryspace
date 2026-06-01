require "./src/cryspace"

# Helper for Butterworth 4 (Standard form for w=1)
# Derived from Transfer Function 1 / (s^4 + 2.613s^3 + 3.414s^2 + 2.613s + 1)
# Converted to Controllable Canonical Form
def butterworth4
  a = [[0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0],
       [0.0, 0.0, 0.0, 1.0],
       [-1.0, -2.6131259, -3.4142136, -2.6131259]].to_tensor
  b = [[0.0], [0.0], [0.0], [1.0]].to_tensor
  c = [[1.0, 0.0, 0.0, 0.0]].to_tensor
  d = [[0.0]].to_tensor
  CrySpace::StateSpace.new(a, b, c, d)
end

m, k, c = 1.0, 10.0, 0.5
a1 = [[0.0, 1.0], [-k/m, -c/m]].to_tensor
b1 = [[0.0], [1/m]].to_tensor
c1 = [[1.0, 0.0]].to_tensor
d1 = [[0.0]].to_tensor
sys1 = CrySpace::StateSpace.new(a1, b1, c1, d1)
sys2 = butterworth4

systems = [{"name": "Oscillator", "sys": sys1}, {"name": "Butterworth4", "sys": sys2}]

puts "--- CRYSTAL FULL PRECISION VALIDATION ---"
systems.each do |s|
  sys = s["sys"]
  puts "\nSystem: #{s["name"]}"
  
  # Format complex poles with full precision
  poles_str = sys.poles.map{|p| sprintf("%.8f + %.8fi", p.real, p.imag)}.join(", ")
  puts "Poles: [#{poles_str}]"
  
  t = Float64Tensor.linear_space(0.0, 10.0, 101)
  u = Float64Tensor.ones([sys.n_inputs, t.size])
  _, _, y = sys.simulate(t, u: u)
  puts "Final Output (t=10s): #{sprintf("%.8f", y[-1, 0].value)}"
  puts "DC Gain: #{sprintf("%.8f", sys.dcgain[0,0].value)}"
end
