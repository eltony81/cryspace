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
      
      t_arr = Array(Float64).new(n_steps)
      x_arr = Array(Float64Tensor).new(n_steps)
      y_arr = Array(Float64Tensor).new(n_steps)
      
      n_steps.times do |i|
        t_arr << i * dt
        x_arr << x
        y = sys.c.matmul(x) + sys.d.matmul(u)
        y_arr << y
        x = sys.a.matmul(x) + sys.b.matmul(u)
      end
      
      {t_arr, x_arr, y_arr}
    end

    def simulate(t_span : Tuple(Float64, Float64), dt : Float64, x0 : Float64Tensor? = nil, u : Float64Tensor? = nil, method = :rk4)
      x_init = x0 || Float64Tensor.zeros([n_states, 1])
      u_val = u || Float64Tensor.zeros([n_inputs, 1])
      
      f = ->(x : Float64Tensor, t : Float64) {
        @a.matmul(x) + @b.matmul(u_val)
      }
      
      res_t, res_x = if method == :rk4
        Solver.rk4(f, x_init, t_span, dt)
      else
        Solver.euler(f, x_init, t_span, dt)
      end

      # Calculate outputs for each state
      res_y = res_x.map do |x_vec|
        @c.matmul(x_vec) + @d.matmul(u_val)
      end

      {res_t, res_x, res_y}
    end

    # Vectorized simulation: optimized for StateSpace systems
    def simulate(t : Float64Tensor, x0 : Float64Tensor? = nil, u : Float64Tensor? = nil, method = :rk4)
      n_steps = t.size
      n_states_count = self.n_states
      n_outputs_count = self.n_outputs
      
      x_matrix = Float64Tensor.new([n_steps, n_states_count])
      y_matrix = Float64Tensor.new([n_steps, n_outputs_count])
      
      x_current = x0.nil? ? Float64Tensor.zeros([n_states_count, 1]) : x0.dup
      u_val = u.nil? ? Float64Tensor.zeros([n_inputs, 1]) : u.dup
      
      n_steps.times do |i|
        # Step 1: Calculate output at current state
        y = @c.matmul(x_current) + @d.matmul(u_val)
        
        # Step 2: Copy current state and output to result matrices
        n_states_count.times do |j|
          val = x_current[j, 0].value
          x_matrix[i, j] = val
        end
        
        n_outputs_count.times do |j|
          val = y[j, 0].value
          y_matrix[i, j] = val
        end
        
        # Step 3: Advance to next state (if not the last step)
        if i < n_steps - 1
          h = t[i + 1].value - t[i].value
          if method == :rk4
            k1 = @a.matmul(x_current) + @b.matmul(u_val)
            k2 = @a.matmul(x_current + k1 * (h / 2.0)) + @b.matmul(u_val)
            k3 = @a.matmul(x_current + k2 * (h / 2.0)) + @b.matmul(u_val)
            k4 = @a.matmul(x_current + k3 * h) + @b.matmul(u_val)
            x_current = x_current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
          else
            k = @a.matmul(x_current) + @b.matmul(u_val)
            x_current = x_current + k * h
          end
        end
      end
      
      {t, x_matrix, y_matrix}
    end

    def impulse_response(n_steps = 100)
      sys = self
      curr_dt = @dt
      unless curr_dt && curr_dt > 0
        sys = self.sample(0.1)
      end
      
      dt = sys.dt.not_nil!
      # For impulse, we set initial state x = B/dt and u = 0 (for discrete approx)
      x = Float64Tensor.zeros([sys.n_states, 1])
      
      t_arr = Array(Float64).new(n_steps)
      x_arr = Array(Float64Tensor).new(n_steps)
      y_arr = Array(Float64Tensor).new(n_steps)
      
      n_steps.times do |i|
        t_arr << i * dt
        u = i == 0 ? (Float64Tensor.ones([sys.n_inputs, 1]) / dt) : Float64Tensor.zeros([sys.n_inputs, 1])
        
        x_arr << x
        y = sys.c.matmul(x) + sys.d.matmul(u)
        y_arr << y
        x = sys.a.matmul(x) + sys.b.matmul(u)
      end
      
      {t_arr, x_arr, y_arr}
    end

    def is_stable?
      p = poles
      n = p.size
      stable = true
      
      n.times do |i|
        val = p[i]
        if @dt.nil? || @dt == 0
          # Continuous: Re(poles) < 0
          stable = false if val.real >= 0
        else
          # Discrete: |poles| < 1
          stable = false if val.abs >= 1.0
        end
      end
      stable
    end

    def ctrb
      # Controllability matrix: [B AB A^2B ... A^(n-1)B]
      n = n_states
      m = n_inputs
      res = Float64Tensor.zeros([n, n * m])
      
      temp = @b.dup
      n.times do |i|
        # res[0...n, (i * m)...((i + 1) * m)] = temp
        n.times do |row|
          m.times do |col|
            res[row, i * m + col] = temp[row, col].value
          end
        end
        temp = @a.matmul(temp)
      end
      res
    end

    def obsv
      # Observability matrix: [C; CA; CA^2; ...; CA^(n-1)]
      n = n_states
      p = n_outputs
      res = Float64Tensor.zeros([n * p, n])
      
      temp = @c.dup
      n.times do |i|
        # res[(i * p)...((i + 1) * p), 0...n] = temp
        p.times do |row|
          n.times do |col|
            res[i * p + row, col] = temp[row, col].value
          end
        end
        temp = temp.matmul(@a)
      end
      res
    end

    def is_controllable?
      _, s, _ = ctrb.svd
      count_nonzero(s) == n_states
    end

    def is_observable?
      _, s, _ = obsv.svd
      count_nonzero(s) == n_states
    end

    private def count_nonzero(s : Float64Tensor, tol = 1e-9)
      count = 0
      s.size.times do |i|
        count += 1 if s[i].value.abs > tol
      end
      count
    end

    def to_transferfunction
      # SISO only for now
      unless n_inputs == 1 && n_outputs == 1
        raise "to_transferfunction only supported for SISO systems"
      end

      n = n_states
      den = Array(Float64).new(n + 1, 0.0)
      den[0] = 1.0
      
      m = Float64Tensor.identity(n)
      n.times do |k|
        index = k + 1
        am = @a.matmul(m)
        trace = 0.0
        n.times { |i| trace += am[i, i].value }
        ak = -trace / index
        den[index] = ak
        m = am + Float64Tensor.identity(n) * ak
      end
      
      num = Array(Float64).new(n + 1, 0.0)
      m = Float64Tensor.identity(n)
      n.times do |k|
        cb = @c.matmul(m).matmul(@b)
        num[k + 1] = cb[0, 0].value
        
        am = @a.matmul(m)
        trace = 0.0
        n.times { |i| trace += am[i, i].value }
        ak = -trace / (k + 1)
        m = am + Float64Tensor.identity(n) * ak
      end
      
      d_val = @d[0, 0].value
      (n + 1).times do |i|
        num[i] += d_val * den[i]
      end
      
      while num.size > 1 && num[0].abs < 1e-12
        num.shift
      end

      TransferFunction.new(num.to_tensor, den.to_tensor, @dt)
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
      # Manual block copy
      n1.times { |r| n1.times { |c| a_cl[r, c] = @a[r, c].value } }
      n2.times { |r| n2.times { |c| a_cl[n1+r, n1+c] = other.a[r, c].value } }
      
      b_cl = Float64Tensor.zeros([n1 + n2, n_inputs])
      n1.times { |r| n_inputs.times { |c| b_cl[r, c] = @b[r, c].value } }
      n2.times { |r| n_inputs.times { |c| b_cl[n1+r, c] = other.b[r, c].value } }
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      n_outputs.times { |r| n1.times { |c| c_cl[r, c] = @c[r, c].value } }
      n_outputs.times { |r| n2.times { |c| c_cl[r, n1+c] = other.c[r, c].value } }
      
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
      bc = @b.matmul(other.c)
      n1.times { |r| n1.times { |c| a_cl[r, c] = @a[r, c].value } }
      n1.times { |r| n2.times { |c| a_cl[r, n1+c] = bc[r, c].value } }
      n2.times { |r| n2.times { |c| a_cl[n1+r, n1+c] = other.a[r, c].value } }
      
      b_cl = Float64Tensor.zeros([n1 + n2, other.n_inputs])
      bd = @b.matmul(other.d)
      n1.times { |r| other.n_inputs.times { |c| b_cl[r, c] = bd[r, c].value } }
      n2.times { |r| other.n_inputs.times { |c| b_cl[n1+r, c] = other.b[r, c].value } }
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      dc = @d.matmul(other.c)
      n_outputs.times { |r| n1.times { |c| c_cl[r, c] = @c[r, c].value } }
      n_outputs.times { |r| n2.times { |c| c_cl[r, n1+c] = dc[r, c].value } }
      
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
