require "num"

module CrySpace
  class TransferFunction
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

    def freqresp(omega : Float64Tensor) : Tensor(Complex, CPU(Complex))
      n_points = omega.size
      res = Tensor(Complex, CPU(Complex)).zeros([n_points])
      n_points.times do |i|
        s = Complex.new(0.0, omega.to_unsafe[i])
        res.to_unsafe[i] = evaluate(s)
      end
      res
    end

    def bode_data(omega : Float64Tensor)
      h = freqresp(omega)
      n_points = omega.size
      mags_db = Array(Float64).new(n_points)
      phases_deg = Array(Float64).new(n_points)
      
      wrap_phase = ->(p : Float64) {
        val = p % (2.0 * Math::PI)
        val -= 2.0 * Math::PI if val > Math::PI
        val += 2.0 * Math::PI if val < -Math::PI
        val
      }

      n_points.times do |i|
        val = h.to_unsafe[i]
        mags_db << 20.0 * Math.log10(val.abs)
        phases_deg << wrap_phase.call(Math.atan2(val.imag, val.real)) * 180.0 / Math::PI
      end
      
      {omega, mags_db, phases_deg}
    end

    def nyquist_data(omega : Float64Tensor)
      h = freqresp(omega)
      n_points = omega.size
      real_parts = Array(Float64).new(n_points)
      imag_parts = Array(Float64).new(n_points)
      
      n_points.times do |i|
        val = h.to_unsafe[i]
        real_parts << val.real
        imag_parts << val.imag
      end
      
      {real_parts, imag_parts}
    end

    def nichols_data(omega : Float64Tensor)
      _, db, deg = bode_data(omega)
      {omega, db, deg}
    end
  end
end
