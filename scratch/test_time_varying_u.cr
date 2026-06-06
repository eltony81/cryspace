require "../src/cryspace"

# Integrator: dx/dt = u, y = x
a = [[0.0]].to_tensor
b = [[1.0]].to_tensor
c = [[1.0]].to_tensor
d = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a, b, c, d)

t = Float64Tensor.linear_space(0.0, 10.0, 11) # t = 0, 1, 2, ..., 10
# Time-varying input: u(t) = t
u = Float64Tensor.new([1, 11])
11.times { |i| u[0, i] = i.to_f }

_, _, y = sys.simulate(t, u: u)
puts "y output:"
p y.to_a
