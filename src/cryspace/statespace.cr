require "num"
require "./statespace_simulation"
require "./statespace_analysis"
require "./statespace_realization"
require "./statespace_connections"
require "./statespace_conversion"
require "./statespace_synthesis"
require "./frd"
require "./descriptor"

module CrySpace
  {% if flag?(:arrow) %}
    alias AnyFloat64Tensor = Tensor(Float64, CPU(Float64)) | Tensor(Float64, ARROW(Float64))
  {% else %}
    alias AnyFloat64Tensor = Tensor(Float64, CPU(Float64))
  {% end %}

  class StateSpace
    property a : Float64Tensor
    property b : Float64Tensor
    property c : Float64Tensor
    property d : Float64Tensor
    property dt : Float64?

    def initialize(@a : Float64Tensor, @b : Float64Tensor, @c : Float64Tensor, @d : Float64Tensor, @dt : Float64? = nil)
      validate_dimensions
    end

    private def validate_dimensions
      n = @a.shape[0]
      m = @b.shape[1]
      p = @c.shape[0]

      unless @a.rank == 2 && @a.shape[0] == @a.shape[1]
        raise ArgumentError.new("A must be a square matrix")
      end

      unless @b.rank == 2 && @b.shape[0] == n
        raise ArgumentError.new("B must have the same number of rows as A")
      end

      unless @c.rank == 2 && @c.shape[1] == n
        raise ArgumentError.new("C must have the same number of columns as A")
      end

      unless @d.rank == 2 && @d.shape[0] == p && @d.shape[1] == m
        raise ArgumentError.new("D must have dimensions (outputs x inputs)")
      end
    end

    def n_states
      @a.shape[0]
    end

    def n_inputs
      @b.shape[1]
    end

    def n_outputs
      @c.shape[0]
    end

    def poles : Array(Complex)
      @a.eigvals_c.to_a
    end

    def dcgain
      if @dt.nil? || @dt == 0
        # Continuous: G(0) = D - C * inv(A) * B
        @d - @c.matmul(@a.solve(@b))
      else
        # Discrete: G(1) = D + C * inv(I - A) * B
        n = n_states
        eye = Float64Tensor.identity(n)
        @d + @c.matmul((eye - @a).solve(@b))
      end
    end

    def self.static_gain(d_matrix : Float64Tensor, dt : Float64? = nil) : StateSpace
      n = 0
      m = d_matrix.shape[1]
      p = d_matrix.shape[0]
      a = Float64Tensor.zeros([0, 0])
      b = Float64Tensor.zeros([0, m])
      c = Float64Tensor.zeros([p, 0])
      StateSpace.new(a, b, c, d_matrix, dt)
    end

    def self.eye(n : Int32, dt : Float64? = nil) : StateSpace
      static_gain(Float64Tensor.identity(n), dt)
    end

    def to_s(io)
      io << "StateSpace system:\n"
      io << "A = " << @a << "\n"
      io << "B = " << @b << "\n"
      io << "C = " << @c << "\n"
      io << "D = " << @d << "\n"
      io << "dt = " << @dt if @dt
    end
  end
end
