require "./src/cryspace"
m, k, c = 1.0, 10.0, 0.5
a = [[0.0, 1.0], [-k/m, -c/m]].to_tensor
b = [[0.0], [1/m]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor
sys = CrySpace::StateSpace.new(a, b, c, d)
t = Float64Tensor.linear_space(0.0, 10.0, 101)
# With input u=1 (standard step response)
u = [[1.0]].to_tensor
_, _, y = sys.simulate(t, u: u)
puts y[-1, 0].value
