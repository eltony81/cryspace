require "num"

module CrySpace
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

    def poles
      @a.eigvals
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

    def feedback(other : StateSpace, sign = -1)
      # Closed loop system with feedback
      # sys1 is self, sys2 is other
      # negative feedback: y = sys1(u - sys2(y))
      
      n1 = n_states
      n2 = other.n_states
      
      # E = inv(I + D2 * D1)
      eye_outputs = Float64Tensor.identity(other.n_inputs)
      e = (eye_outputs + other.d.matmul(@d)).inv
      
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      a_cl[0...n1, 0...n1] = @a - @b.matmul(e).matmul(other.d).matmul(@c)
      a_cl[0...n1, n1...] = -@b.matmul(e).matmul(other.c)
      a_cl[n1..., 0...n1] = other.b.matmul(@c - @d.matmul(e).matmul(other.d).matmul(@c))
      a_cl[n1..., n1...] = other.a - other.b.matmul(@d).matmul(e).matmul(other.c)
      
      b_cl = Float64Tensor.zeros([n1 + n2, n_inputs])
      b_cl[0...n1, 0...] = @b.matmul(e)
      b_cl[n1..., 0...] = other.b.matmul(@d).matmul(e)
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      c_cl[0..., 0...n1] = @c - @d.matmul(e).matmul(other.d).matmul(@c)
      c_cl[0..., n1...] = -@d.matmul(e).matmul(other.c)
      
      d_cl = @d.matmul(e)
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
    end

    def feedback(k : Float64Tensor, sign = -1)
      # feedback with static gain K
      # E = inv(I + K * D)
      eye_k = Float64Tensor.identity(k.shape[0])
      e = (eye_k + k.matmul(@d)).inv
      
      a_cl = @a - @b.matmul(e).matmul(k).matmul(@c)
      b_cl = @b.matmul(e)
      
      eye_d = Float64Tensor.identity(n_outputs)
      inv_idk = (eye_d + @d.matmul(k)).inv
      
      c_cl = inv_idk.matmul(@c)
      d_cl = inv_idk.matmul(@d)
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
    end

    def sample(dt : Float64)
      curr_dt = @dt
      if curr_dt && curr_dt > 0
        raise "System is already discrete"
      end
      
      ad = expm(@a * dt)
      
      # Bd = dt * (I + A*dt/2! + A^2*dt^2/3! + ...) * B
      n = n_states
      bd_term = Float64Tensor.identity(n)
      sum_term = Float64Tensor.identity(n)
      (1..15).each do |i|
        bd_term = bd_term.matmul(@a * dt) / (i + 1).to_f
        sum_term = sum_term + bd_term
      end
      bd = sum_term.matmul(@b) * dt
      
      StateSpace.new(ad, bd, @c, @d, dt)
    end

    def step_response(n_steps = 100)
      sys = self
      curr_dt = @dt
      unless curr_dt && curr_dt > 0
        sys = self.sample(0.1)
      end
      
      dt = sys.dt.not_nil!
      x = Float64Tensor.zeros([sys.n_states, 1])
      u = Float64Tensor.ones([sys.n_inputs, 1])
      
      t = [] of Float64
      x_out = [] of Float64Tensor
      y_out = [] of Float64Tensor
      
      n_steps.times do |i|
        t << i * dt
        x_out << x
        y = sys.c.matmul(x) + sys.d.matmul(u)
        y_out << y
        x = sys.a.matmul(x) + sys.b.matmul(u)
      end
      
      {t, x_out, y_out}
    end

    def simulate(t_span : Tuple(Float64, Float64), dt : Float64, x0 : Float64Tensor? = nil, u : Float64Tensor? = nil, method = :rk4)
      x_init = x0 || Float64Tensor.zeros([n_states, 1])
      u_val = u || Float64Tensor.zeros([n_inputs, 1])
      
      f = ->(x : Float64Tensor, t : Float64) {
        @a.matmul(x) + @b.matmul(u_val)
      }
      
      if method == :rk4
        Solver.rk4(f, x_init, t_span, dt)
      else
        Solver.euler(f, x_init, t_span, dt)
      end
    end

    private def expm(m : Float64Tensor, order = 15)
      n = m.shape[0]
      res = Float64Tensor.identity(n)
      term = Float64Tensor.identity(n)
      (1..order).each do |i|
        term = term.matmul(m) / i.to_f
        res = res + term
      end
      res
    end

    def +(other : StateSpace)
      unless n_inputs == other.n_inputs && n_outputs == other.n_outputs
        raise ArgumentError.new("Systems must have same number of inputs and outputs for parallel connection")
      end

      n1 = n_states
      n2 = other.n_states
      
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      a_cl[0...n1, 0...n1] = @a
      a_cl[n1..., n1...] = other.a
      
      b_cl = Float64Tensor.zeros([n1 + n2, n_inputs])
      b_cl[0...n1, 0...] = @b
      b_cl[n1..., 0...] = other.b
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      c_cl[0..., 0...n1] = @c
      c_cl[0..., n1...] = other.c
      
      d_cl = @d + other.d
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
    end

    def *(other : StateSpace)
      unless n_inputs == other.n_outputs
        raise ArgumentError.new("System 1 inputs must match System 2 outputs for series connection")
      end

      n1 = n_states
      n2 = other.n_states
      
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      a_cl[0...n1, 0...n1] = @a
      a_cl[0...n1, n1...] = @b.matmul(other.c)
      a_cl[n1..., n1...] = other.a
      
      b_cl = Float64Tensor.zeros([n1 + n2, other.n_inputs])
      b_cl[0...n1, 0...] = @b.matmul(other.d)
      b_cl[n1..., 0...] = other.b
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      c_cl[0..., 0...n1] = @c
      c_cl[0..., n1...] = @d.matmul(other.c)
      
      d_cl = @d.matmul(other.d)
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
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
