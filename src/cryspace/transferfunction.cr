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
      return Array(Complex).new if n <= 0
      
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

    # Computes the Sensitivity function S = 1 / (1 + G*K)
    def sensitivity(k : TransferFunction) : TransferFunction
      one = TransferFunction.new([1.0].to_tensor, [1.0].to_tensor, @dt)
      one.feedback(self * k)
    end

    # Computes the Complementary Sensitivity function T = G*K / (1 + G*K)
    def complementary_sensitivity(k : TransferFunction) : TransferFunction
      one = TransferFunction.new([1.0].to_tensor, [1.0].to_tensor, @dt)
      (self * k).feedback(one)
    end

    # Transforms a lowpass filter to a highpass filter with given cutoff frequency.
    def lowpass_to_highpass(cutoff_frequency : Float64) : TransferFunction
      n = @den.size - 1
      num_padded = Array(Float64).new(n + 1, 0.0)
      offset = n + 1 - @num.size
      @num.size.times do |i|
        num_padded[i + offset] = @num[i].value
      end
      
      new_num = Array(Float64).new(n + 1, 0.0)
      new_den = Array(Float64).new(n + 1, 0.0)
      
      (n + 1).times do |i|
        new_num[n - i] = num_padded[i] * (cutoff_frequency ** (n - i))
        new_den[n - i] = @den[i].value * (cutoff_frequency ** (n - i))
      end
      
      TransferFunction.new(new_num.to_tensor, new_den.to_tensor, @dt)
    end

    private def complex_sqrt(z : Complex) : Complex
      r = z.abs
      theta = Math.atan2(z.imag, z.real)
      sqrt_r = Math.sqrt(r)
      Complex.new(sqrt_r * Math.cos(theta / 2.0), sqrt_r * Math.sin(theta / 2.0))
    end

    # Transforms a lowpass filter to a bandpass filter with given center frequency and bandwidth.
    def lowpass_to_bandpass(center_frequency : Float64, bandwidth : Float64) : TransferFunction
      p_list = poles
      z_list = zeros
      
      w0_sq_4 = 4.0 * (center_frequency ** 2)
      
      new_poles = Array(Complex).new
      p_list.each do |p|
        val = (bandwidth * p)
        sqrt_term = complex_sqrt(val * val - w0_sq_4)
        new_poles << (val + sqrt_term) / 2.0
        new_poles << (val - sqrt_term) / 2.0
      end
      
      new_zeros = Array(Complex).new
      z_list.each do |z|
        val = (bandwidth * z)
        sqrt_term = complex_sqrt(val * val - w0_sq_4)
        new_zeros << (val + sqrt_term) / 2.0
        new_zeros << (val - sqrt_term) / 2.0
      end
      
      num_zeros_at_origin = p_list.size - z_list.size
      num_zeros_at_origin.times do
        new_zeros << Complex.new(0.0, 0.0)
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
      
      num_new = reconstruct.call(new_zeros)
      den_new = reconstruct.call(new_poles)
      
      gain_factor = bandwidth ** (p_list.size - z_list.size)
      num_new = num_new * gain_factor
      
      TransferFunction.new(num_new, den_new, @dt)
    end

    # Discretizes a continuous TransferFunction directly.
    def to_discrete(dt : Float64, method : Symbol = :zoh) : TransferFunction
      if method == :matched
        s_poles = poles
        s_zeros = zeros
        
        # Complex exponential mapping: e^(s * dt)
        complex_exp = ->(c : Complex) {
          r = Math.exp(c.real * dt)
          theta = c.imag * dt
          Complex.new(r * Math.cos(theta), r * Math.sin(theta))
        }
        
        z_poles = s_poles.map { |p| complex_exp.call(p) }
        z_zeros = s_zeros.map { |z| complex_exp.call(z) }
        
        n_poles = s_poles.size
        n_zeros = s_zeros.size
        if n_poles > n_zeros
          num_inf_zeros = [n_poles - n_zeros - 1, 0].max
          num_inf_zeros.times do
            z_zeros << Complex.new(-1.0, 0.0)
          end
        end
        
        den_d_c = TransferFunction.reconstruct_poly(z_poles)
        num_d_c = TransferFunction.reconstruct_poly(z_zeros)
        
        has_dc_pole_or_zero = s_poles.any? { |p| p.abs < 1e-5 } || s_zeros.any? { |z| z.abs < 1e-5 }
        s_match = has_dc_pole_or_zero ? Complex.new(0.0, 0.1 * Math::PI / dt) : Complex.new(0.0, 0.0)
        z_match = complex_exp.call(s_match)
        
        g_cont = evaluate(s_match)
        
        num_val_d = Complex.new(0.0, 0.0)
        num_d_c.size.times do |i|
          num_val_d += Complex.new(num_d_c[i].value, 0.0) * complex_power(z_match, num_d_c.size - 1 - i)
        end
        den_val_d = Complex.new(0.0, 0.0)
        den_d_c.size.times do |i|
          den_val_d += Complex.new(den_d_c[i].value, 0.0) * complex_power(z_match, den_d_c.size - 1 - i)
        end
        g_disc_raw = num_val_d / den_val_d
        
        kd = (g_cont / g_disc_raw).real
        num_d = num_d_c * kd
        TransferFunction.new(num_d, den_d_c, dt)
      else
        to_statespace.sample(dt, method).to_transferfunction
      end
    end

    private def complex_power(base : Complex, power : Int32) : Complex
      res = Complex.new(1.0, 0.0)
      power.times { res *= base }
      res
    end

    # Evaluates the transfer function at a complex frequency s.
    def evaluate(s : Complex) : Complex
      n_num = @num.size
      n_den = @den.size
      num_val = Complex.new(0.0, 0.0)
      n_num.times do |i|
        num_val += Complex.new(@num[i].value, 0.0) * complex_power(s, n_num - 1 - i)
      end
      den_val = Complex.new(0.0, 0.0)
      n_den.times do |i|
        den_val += Complex.new(@den[i].value, 0.0) * complex_power(s, n_den - 1 - i)
      end
      num_val / den_val
    end

    # Reconstructs polynomial coefficients from its roots.
    def self.reconstruct_poly(roots : Array(Complex)) : Float64Tensor
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
    end

    # Designs a phase lead or lag compensator: Gc(s) = gain * (s + zero) / (s + pole).
    def self.leadlag(zero : Float64, pole : Float64, gain : Float64 = 1.0) : TransferFunction
      TransferFunction.new([gain, gain * zero].to_tensor, [1.0, pole].to_tensor)
    end

    # Generates a standard loop shaping weighting filter: W(s) = (s/high + middle) / (s + middle * low).
    def self.makeweight(low : Float64, middle : Float64, high : Float64) : TransferFunction
      TransferFunction.new([1.0 / high, middle].to_tensor, [1.0, middle * low].to_tensor)
    end

    # Converts a discrete TransferFunction back to continuous.
    def to_continuous : TransferFunction
      to_statespace.to_continuous.to_transferfunction
    end

    # Designs an analog lowpass Chebyshev Type I filter of given order, passband ripple rp, and cutoff frequency Wn.
    def self.cheby1(order : Int32, rp : Float64, wn : Float64) : TransferFunction
      raise ArgumentError.new("Order must be at least 1") if order < 1
      raise ArgumentError.new("Cutoff frequency Wn must be positive") if wn <= 0.0
      
      eps = Math.sqrt(10.0 ** (rp / 10.0) - 1.0)
      mu = Math.asinh(1.0 / eps) / order
      poles = Array(Complex).new(order)
      
      order.times do |k|
        theta = Math::PI * (2 * (k + 1) - 1) / (2 * order)
        s_real = -wn * Math.sinh(mu) * Math.sin(theta)
        s_imag = wn * Math.cosh(mu) * Math.cos(theta)
        poles << Complex.new(s_real, s_imag)
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
      dc_target = order.odd? ? 1.0 : 1.0 / Math.sqrt(1.0 + eps * eps)
      num_val = den.to_a.last * dc_target
      num = [num_val].to_tensor
      
      TransferFunction.new(num, den)
    end

    # Designs a digital lowpass Butterworth filter using bilinear discretization.
    def self.butter_digital(order : Int32, wn : Float64, dt : Float64) : TransferFunction
      butter(order, wn).to_discrete(dt, :tustin)
    end

    # Computes the Routh-Hurwitz stability table for the denominator polynomial of a continuous system.
    def routh_hurwitz : NamedTuple(stable: Bool, table: Array(Array(Float64)))
      coeffs = @den.to_a
      n = coeffs.size - 1
      raise ArgumentError.new("System denominator must have degree >= 1") if n < 1
      
      n_cols = (n + 2) // 2
      table = Array(Array(Float64)).new(n + 1) { Array(Float64).new(n_cols, 0.0) }
      
      n_cols.times do |i|
        idx = 2 * i
        table[0][i] = idx < coeffs.size ? coeffs[idx] : 0.0
      end
      
      n_cols.times do |i|
        idx = 2 * i + 1
        table[1][i] = idx < coeffs.size ? coeffs[idx] : 0.0
      end
      
      (2..n).each do |r|
        n_cols.times do |c|
          val_prev = table[r-1][0]
          if val_prev.abs < 1e-12
            val_prev = 1e-12
          end
          
          term1 = val_prev * (c + 1 < n_cols ? table[r-2][c+1] : 0.0)
          term2 = table[r-2][0] * (c + 1 < n_cols ? table[r-1][c+1] : 0.0)
          table[r][c] = (term1 - term2) / val_prev
        end
      end
      
      sign = table[0][0] <=> 0.0
      stable = true
      (0..n).each do |r|
        if (table[r][0] <=> 0.0) != sign || table[r][0].abs < 1e-9
          stable = false
        end
      end
      
      {stable: stable, table: table}
    end

    # Computes the Jury stability table for discrete systems.
    def jury_test : NamedTuple(stable: Bool, table: Array(Array(Float64)))
      coeffs = @den.to_a
      n = coeffs.size - 1
      raise ArgumentError.new("System denominator must have degree >= 1") if n < 1
      
      rows = Array(Array(Float64)).new
      rows << coeffs.dup
      rows << coeffs.reverse
      
      (1..n-1).each do |k|
        prev_row = rows[rows.size - 2]
        
        len = n - k + 1
        next_row = Array(Float64).new(len, 0.0)
        
        a0 = prev_row[0]
        an = prev_row[prev_row.size - 1]
        
        len.times do |i|
          next_row[i] = a0 * prev_row[i] - an * prev_row[prev_row.size - 1 - i]
        end
        
        rows << next_row.dup
        rows << next_row.reverse
      end
      
      stable = true
      p_1 = 0.0
      p_minus_1 = 0.0
      coeffs.each_with_index do |c, idx|
        p_1 += c
        p_minus_1 += c * (idx.even? ? 1.0 : -1.0)
      end
      
      if p_1 <= 0.0
        stable = false
      end
      
      if ((-1.0)**n * p_minus_1) <= 0.0
        stable = false
      end
      
      (0..n-2).each do |k|
        row_idx = 2 * k
        row = rows[row_idx]
        if row[0].abs <= row[row.size - 1].abs
          stable = false
        end
      end
      
      {stable: stable, table: rows}
    end

    # Designs a continuous-time notch filter G(s) = (s^2 + w0^2) / (s^2 + 2*zeta*w0*s + w0^2).
    def self.notch(w0 : Float64, zeta : Float64) : TransferFunction
      raise ArgumentError.new("Center frequency w0 must be positive") if w0 <= 0.0
      raise ArgumentError.new("Damping ratio zeta must be positive") if zeta <= 0.0
      
      num = [1.0, 0.0, w0 * w0].to_tensor
      den = [1.0, 2.0 * zeta * w0, w0 * w0].to_tensor
      TransferFunction.new(num, den)
    end
  end
end
