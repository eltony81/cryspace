require "num"

module CrySpace
  class TransferFunction
    property num : Float64Tensor
    property den : Float64Tensor
    property dt : Float64?

    def initialize(@num : Float64Tensor, @den : Float64Tensor, @dt : Float64? = nil)
      # Normalize denominator
      if @den[0].value != 1.0
        alpha = @den[0].value
        @num = @num / alpha
        @den = @den / alpha
      end
    end

    # Computes a Pade approximation of a time delay.
    # Returns a TransferFunction approximation of e^(-delay * s).
    def self.pade(delay : Float64, order : Int32 = 1) : TransferFunction
      if delay <= 0.0
        return TransferFunction.new([1.0].to_tensor, [1.0].to_tensor)
      end

      # Coefficients are calculated using standard recurrence formula:
      # c_k = (order)! * (2*order - k)! / ( k! * (order - k)! * (2*order)! )
      # num_k = (-delay)^k * c_k
      # den_k = (delay)^k * c_k
      
      factorial = ->(n : Int32) {
        val = 1.0
        (1..n).each { |i| val *= i }
        val
      }

      num_arr = Array(Float64).new(order + 1, 0.0)
      den_arr = Array(Float64).new(order + 1, 0.0)

      (order + 1).times do |k|
        # Coeff index corresponding to s^(order - k)
        # Power is (order - k)
        power = order - k
        c_val = (factorial.call(order) * factorial.call(2 * order - power)) /
                (factorial.call(power) * factorial.call(order - power) * factorial.call(2 * order))
        
        num_arr[k] = ((-delay) ** power) * c_val
        den_arr[k] = ((delay) ** power) * c_val
      end

      TransferFunction.new(num_arr.to_tensor, den_arr.to_tensor)
    end
    def poles : Array(Complex)
      roots(@den)
    end

    def zeros : Array(Complex)
      roots(@num)
    end

    private def roots(poly : Float64Tensor) : Array(Complex)
      # Roots of polynomial using companion matrix eigenvalues
      # Assumes poly[0] is the coefficient of highest power
      n = poly.size - 1
      return [Complex.new(0.0, 0.0)] if n <= 0
      
      # Normalize
      p = poly / poly[0].value
      
      companion = Float64Tensor.zeros([n, n])
      (n - 1).times do |i|
        companion[i + 1, i] = 1.0
      end
      n.times do |i|
        # coefficients are in order [p0, p1, ..., pn]
        # characteristic eq: s^n + a1 s^(n-1) + ... + an = 0
        # companion matrix first row: [-a1, -a2, ..., -an]
        companion[0, i] = -p[i + 1].value
      end
      companion.eigvals_c.to_a
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
  # G_cl = G1 / (1 - sign * G1*G2)
  # sign = -1 (Negative feedback): (N1*D2) / (D1*D2 + N1*N2)
  # sign = 1 (Positive feedback): (N1*D2) / (D1*D2 - N1*N2)
  num_new = poly_mul(@num, other.den)
  
  prod = poly_mul(@num, other.num)
  if sign == 1
    # Subtract prod
    neg_prod = prod * -1.0
    den_new = poly_add(poly_mul(@den, other.den), neg_prod)
  else
    den_new = poly_add(poly_mul(@den, other.den), prod)
  end

  TransferFunction.new(num_new, den_new, @dt)
end

# Minimal realization for transfer function: performs pole-zero cancellation
def minreal(tol = 1e-3) : TransferFunction
  p_list = poles
  z_list = zeros
  
  cancelled_poles = Array(Complex).new
  cancelled_zeros = Array(Complex).new

  keep_poles = p_list.dup
  keep_zeros = z_list.dup

  p_list.each do |p|
    z_list.each do |z|
      if (p - z).abs < tol && keep_poles.includes?(p) && keep_zeros.includes?(z)
        keep_poles.delete(p)
        keep_zeros.delete(z)
        cancelled_poles << p
        cancelled_zeros << z
      end
    end
  end

  # Reconstruct polynomial from remaining roots
  # poly = (s - r1)(s - r2)...
  reconstruct = ->(roots : Array(Complex)) {
    return [[1.0]].to_tensor if roots.empty?
    # Start with poly = [1.0]
    poly = [1.0]
    roots.each do |r|
      # multiply poly by (s - r)
      # if r is complex, we might have complex coefficients, but control system polynomials are real.
      # To keep it robust, we do polynomial multiplication with complex numbers, then take real parts
      next_poly = Array(Complex).new(poly.size + 1, Complex.new(0.0, 0.0))
      poly.size.times do |i|
        # s * poly[i] -> index i
        next_poly[i] += Complex.new(poly[i], 0.0)
        # -r * poly[i] -> index i+1
        next_poly[i + 1] += Complex.new(poly[i], 0.0) * -r
      end
      poly = next_poly.map(&.real)
    end
    poly.to_tensor
  }

  num_new = reconstruct.call(keep_zeros)
  den_new = reconstruct.call(keep_poles)

  # Scale back to match the original DC gain / high frequency gain scale if needed
  # We can normalize the lead coefficient of denominator to 1.0 (done by constructor)
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

    def self.lead_compensator(gain : Float64, zero : Float64, pole : Float64) : TransferFunction
      raise ArgumentError.new("Zero must be less than pole for a lead compensator") if zero >= pole
      TransferFunction.new([gain, gain * zero].to_tensor, [1.0, pole].to_tensor)
    end

    def self.lag_compensator(gain : Float64, zero : Float64, pole : Float64) : TransferFunction
      raise ArgumentError.new("Zero must be greater than pole for a lag compensator") if zero <= pole
      TransferFunction.new([gain, gain * zero].to_tensor, [1.0, pole].to_tensor)
    end

    # Designs an analog lowpass Butterworth filter of a given order and cutoff frequency Wn.
    def self.butter(order : Int32, wn : Float64) : TransferFunction
      raise ArgumentError.new("Butterworth filter order must be at least 1") if order < 1
      raise ArgumentError.new("Cutoff frequency Wn must be positive") if wn <= 0.0
      
      poles = Array(Complex).new(order)
      order.times do |k|
        theta = Math::PI * (2 * (k + 1) + order - 1) / (2 * order)
        poles << Complex.new(wn * Math.cos(theta), wn * Math.sin(theta))
      end
      
      reconstruct = ->(roots : Array(Complex)) {
        poly = [Complex.new(1.0, 0.0)]
        roots.each do |r|
          next_poly = Array(Complex).new(poly.size + 1, Complex.new(0.0, 0.0))
          poly.size.times do |i|
            next_poly[i] += poly[i]
            next_poly[i + 1] += poly[i] * -r
          end
          poly = next_poly
        end
        poly.map(&.real).to_tensor
      }
      
      den = reconstruct.call(poles)
      num_val = wn ** order
      num = [num_val].to_tensor
      
      TransferFunction.new(num, den)
    end
  end
end
