require "num"

module CrySpace
  class TransferFunction
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

    def -(other : TransferFunction)
      neg_num = other.num * -1.0
      neg_tf = TransferFunction.new(neg_num, other.den, other.dt)
      self + neg_tf
    end

    def /(other : TransferFunction)
      inv_tf = TransferFunction.new(other.den, other.num, other.dt)
      self * inv_tf
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
  end
end
