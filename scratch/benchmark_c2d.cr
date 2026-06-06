require "../src/cryspace"
require "benchmark"

# Set up a random state-space system
n = 10  # 10 states
m = 2   # 2 inputs

Num::Rand.set_seed(0)
a = Tensor.random(-1.0...1.0, [n, n])
b = Tensor.random(-1.0...1.0, [n, m])
c = Tensor.random(-1.0...1.0, [1, n])
d = Tensor.random(-1.0...1.0, [1, m])

sys = CrySpace::StateSpace.new(a, b, c, d)
dt = 0.05

# Method 1: Original Taylor Series
def sample_taylor(sys, dt)
  a = sys.a
  b = sys.b
  c = sys.c
  d = sys.d
  n = sys.n_states
  
  ad = (a * dt).expm
  
  bd_term = Float64Tensor.identity(n)
  sum_term = Float64Tensor.identity(n)
  (1..15).each do |i|
    bd_term = bd_term.matmul(a * dt) / (i + 1).to_f
    sum_term = sum_term + bd_term
  end
  bd = sum_term.matmul(b) * dt
  {ad, bd}
end

# Method 2: Block Matrix Exponentiation
def sample_block_expm(sys, dt)
  a = sys.a
  b = sys.b
  n = sys.n_states
  m = sys.n_inputs
  
  # Construct block matrix of size (n+m) x (n+m)
  block = Float64Tensor.zeros([n + m, n + m])
  
  # Copy a into top-left
  n.times do |i|
    n.times do |j|
      block[i, j] = a[i, j].value
    end
  end
  
  # Copy b into top-right
  n.times do |i|
    m.times do |j|
      block[i, n + j] = b[i, j].value
    end
  end
  
  scaled_block = block * dt
  e = scaled_block.expm
  
  # Extract ad and bd
  ad = Float64Tensor.zeros([n, n])
  bd = Float64Tensor.zeros([n, m])
  
  n.times do |i|
    n.times do |j|
      ad[i, j] = e[i, j].value
    end
  end
  
  n.times do |i|
    m.times do |j|
      bd[i, j] = e[i, n + j].value
    end
  end
  
  {ad, bd}
end

# Verify correctness
ad1, bd1 = sample_taylor(sys, dt)
ad2, bd2 = sample_block_expm(sys, dt)

puts "Ad maximum difference: #{(ad1 - ad2).map(&.abs).max}"
puts "Bd maximum difference: #{(bd1 - bd2).map(&.abs).max}"

Benchmark.ips do |x|
  x.report("Taylor series") { sample_taylor(sys, dt) }
  x.report("Block Matrix Expm") { sample_block_expm(sys, dt) }
end
