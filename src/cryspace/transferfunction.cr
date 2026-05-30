require "num"

module CrySpace
  class TransferFunction
    property num : Float64Tensor
    property den : Float64Tensor
    property dt : Float64?

    def initialize(@num : Float64Tensor, @den : Float64Tensor, @dt : Float64? = nil)
      # Normalize denominator
      if @den[0] != 1.0
        alpha = @den[0]
        @num = @num / alpha
        @den = @den / alpha
      end
    end
def poles
  roots(@den)
end

def zeros
  roots(@num)
end

private def roots(poly : Float64Tensor)
  # Roots of polynomial using companion matrix eigenvalues
  # Assumes poly[0] is the coefficient of highest power
  n = poly.size - 1
  return Float64Tensor.new([0]) if n <= 0

  # Normalize
  p = poly / poly[0].value

  companion = Float64Tensor.zeros([n, n])
  (n - 1).times do |i|
    companion[i + 1, i] = 1.0
  end
  n.times do |i|
    companion[0, i] = -p[i + 1].value
  end
  companion.eigvals
end

def +(other : TransferFunction)
  # G1 + G2 = (N1*D2 + N2*D1) / (D1*D2)
  n1 = poly_mul(@num, other.den)
  n2 = poly_mul(other.num, @den)

  num_new = poly_add(n1, n2)
  den_new = poly_mul(@den, other.den)

  TransferFunction.new(num_new, den_new, @dt)
end

def *(other : TransferFunction)
  # G1 * G2 = (N1*N2) / (D1*D2)
  num_new = poly_mul(@num, other.num)
  den_new = poly_mul(@den, other.den)

  TransferFunction.new(num_new, den_new, @dt)
end

def feedback(other : TransferFunction, sign = -1)
  # G_cl = G1 / (1 + G1*G2) = (N1*D2) / (D1*D2 + N1*N2)
  num_new = poly_mul(@num, other.den)
  den_new = poly_add(poly_mul(@den, other.den), poly_mul(@num, other.num))

  TransferFunction.new(num_new, den_new, @dt)
end

private def poly_mul(p1 : Float64Tensor, p2 : Float64Tensor)
  n1 = p1.size
  n2 = p2.size
  res_size = n1 + n2 - 1
  res = Array(Float64).new(res_size, 0.0)

  n1.times do |i|
    n2.times do |j|
      res[i + j] += p1[i].value * p2[j].value
    end
  end
  res.to_tensor
end

private def poly_add(p1 : Float64Tensor, p2 : Float64Tensor)
  n1 = p1.size
  n2 = p2.size
  res_size = {n1, n2}.max
  res = Array(Float64).new(res_size, 0.0)

  res_size.times do |i|
    idx1 = n1 - 1 - i
    idx2 = n2 - 1 - i
    val1 = idx1 >= 0 ? p1[idx1].value : 0.0
    val2 = idx2 >= 0 ? p2[idx2].value : 0.0
    res[res_size - 1 - i] = val1 + val2
  end
  res.to_tensor
end

def to_statespace
...

      # Convert to StateSpace using Controllable Canonical Form
      # Assumes SISO system
      n = @den.size - 1
      if n == 0
        return StateSpace.new(
          Float64Tensor.zeros([0, 0]),
          Float64Tensor.zeros([0, 1]),
          Float64Tensor.zeros([1, 0]),
          [[@num[0].value]].to_tensor,
          @dt
        )
      end

      a = Float64Tensor.zeros([n, n])
      n.times do |i|
        a[0, i] = -@den[i + 1]
      end
      (n - 1).times do |i|
        a[i + 1, i] = 1.0
      end

      b = Float64Tensor.zeros([n, 1])
      b[0, 0] = 1.0

      # Pad numerator with zeros if needed
      num_padded = Float64Tensor.zeros([n + 1])
      offset = n + 1 - @num.size
      @num.size.times do |i|
        num_padded[i + offset] = @num[i]
      end

      b0 = num_padded[0].value
      c = Float64Tensor.zeros([1, n])
      n.times do |i|
        c[0, i] = num_padded[i + 1].value - b0 * @den[i + 1].value
      end

      d = [[b0]].to_tensor

      StateSpace.new(a, b, c, d, @dt)
    end

    def to_s(io)
      io << "TransferFunction system:\n"
      io << "num = " << @num << "\n"
      io << "den = " << @den << "\n"
      io << "dt = " << @dt if @dt
    end
    
    # TODO: to_statespace, +, *, feedback
  end
end
