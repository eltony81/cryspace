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
      # Roots of denominator
      # In control theory, we often use companion matrix eigenvalues
      n = @den.size - 1
      return Float64Tensor.new([0]) if n == 0
      
      companion = Float64Tensor.zeros([n, n])
      (n - 1).times do |i|
        companion[i + 1, i] = 1.0
      end
      n.times do |i|
        companion[0, i] = -@den[i + 1]
      end
      companion.eigvals
    end

    def to_statespace
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
