require "../src/cryspace"

# Double integrator
a = [[0.0, 1.0], [0.0, 0.0]].to_tensor
b = [[0.0], [1.0]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a, b, c, d)
sys_dual = CrySpace::StateSpace.new(sys.a.transpose, sys.c.transpose, sys.b.transpose, sys.d.transpose)

# Let's inspect phi(A) inside sys_dual.acker
# phi = (A + 4I)(A + 5I) = A^2 + 9A + 20I
# A_dual^2 = [0 0; 1 0] * [0 0; 1 0] = [0 0; 0 0]
# 9*A_dual = [0 0; 9 0]
# 20*I = [20 0; 0 20]
# phi = [20 0; 9 20]

# Let's see: last_row = [0, 1]
# last_row * phi = [0, 1] * [20 0; 9 20] = [9, 20]

# Why does K print [[0, 20]]?
# Let's check:
a_c = Tensor(Complex, CPU(Complex)).zeros([2, 2])
2.times do |i|
  2.times do |j|
    a_c.to_unsafe[i * 2 + j] = Complex.new(sys_dual.a.to_unsafe[i * 2 + j], 0.0)
  end
end
puts "a_c: #{a_c}"

phi = Tensor(Complex, CPU(Complex)).eye(2)
eye_c = Tensor(Complex, CPU(Complex)).eye(2)

[-4.0, -5.0].each do |p|
  c_pole = Complex.new(p, 0.0)
  term = a_c - eye_c * c_pole
  puts "term for pole #{p}: #{term}"
  phi = phi.matmul(term)
end
puts "phi: #{phi}"
