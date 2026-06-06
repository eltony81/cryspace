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

    def sample(dt : Float64, method : Symbol = :zoh, alpha : Float64 = 0.5)
      curr_dt = @dt
      if curr_dt && curr_dt > 0
        raise "System is already discrete"
      end
      
      if method == :zoh
        ad = (@a * dt).expm
        
        n = n_states
        bd_term = Float64Tensor.identity(n)
        sum_term = Float64Tensor.identity(n)
        (1..15).each do |i|
          bd_term = bd_term.matmul(@a * dt) / (i + 1).to_f
          sum_term = sum_term + bd_term
        end
        bd = sum_term.matmul(@b) * dt
        
        StateSpace.new(ad, bd, @c, @d, dt)
      elsif method == :gbt || method == :bilinear || method == :tustin
        gbt_alpha = (method == :bilinear || method == :tustin) ? 0.5 : alpha
        
        n = n_states
        identity_n = Float64Tensor.identity(n)
        
        # M = (I - alpha * dt * A)^-1
        m = (identity_n - @a * (gbt_alpha * dt)).inv
        
        # Ad = M * (I + (1 - alpha) * dt * A)
        ad = m.matmul(identity_n + @a * ((1.0 - gbt_alpha) * dt))
        
        # Bd = M * B * dt
        bd = m.matmul(@b) * dt
        
        # Cd = C * M
        cd = @c.matmul(m)
        
        # Dd = D + alpha * dt * C * M * B
        dd = @d + @c.matmul(m).matmul(@b) * (gbt_alpha * dt)
        
        StateSpace.new(ad, bd, cd, dd, dt)
      else
        raise ArgumentError.new("Discretization method must be :zoh, :gbt, :bilinear, or :tustin")
      end
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
      n_inputs_count = self.n_inputs
      
      x_matrix = Float64Tensor.new([n_steps, n_states_count])
      y_matrix = Float64Tensor.new([n_steps, n_outputs_count])
      
      x_current = x0.nil? ? Float64Tensor.zeros([n_states_count, 1]) : x0.dup
      u_val = u.nil? ? Float64Tensor.zeros([n_inputs_count, 1]) : (u.is_c_contiguous ? u : u.dup(Num::RowMajor))
      u_current = Float64Tensor.zeros([n_inputs_count, 1])
      
      n_steps.times do |i|
        if u_val.shape[1] > 1
          n_inputs_count.times do |j|
            u_current.to_unsafe[j] = u_val.to_unsafe[j * n_steps + i]
          end
        else
          u_current = u_val
        end

        # Step 1: Calculate output at current state
        y = @c.matmul(x_current) + @d.matmul(u_current)
        
        # Step 2: Copy current state and output to result matrices using pointers
        n_states_count.times do |j|
          x_matrix.to_unsafe[i * n_states_count + j] = x_current.to_unsafe[j]
        end
        
        n_outputs_count.times do |j|
          y_matrix.to_unsafe[i * n_outputs_count + j] = y.to_unsafe[j]
        end
        
        # Step 3: Advance to next state (if not the last step)
        if i < n_steps - 1
          h = t.to_unsafe[i + 1] - t.to_unsafe[i]
          if method == :rk4
            k1 = @a.matmul(x_current) + @b.matmul(u_current)
            k2 = @a.matmul(x_current + k1 * (h / 2.0)) + @b.matmul(u_current)
            k3 = @a.matmul(x_current + k2 * (h / 2.0)) + @b.matmul(u_current)
            k4 = @a.matmul(x_current + k3 * h) + @b.matmul(u_current)
            x_current = x_current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
          else
            k = @a.matmul(x_current) + @b.matmul(u_current)
            x_current = x_current + k * h
          end
        end
      end
      
      {t, x_matrix, y_matrix}
    end

    # Simulates impulse response of discrete or continuous (sampled) system.
    def impulse_response(n_steps = 100)
      sys = self
      curr_dt = @dt
      unless curr_dt && curr_dt > 0
        sys = self.sample(0.1)
      end
      
      dt = sys.dt.not_nil!
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

    # Simulates unforced free response with non-zero initial state x0.
    def initial_response(x0 : Float64Tensor, n_steps = 100)
      sys = self
      curr_dt = @dt
      unless curr_dt && curr_dt > 0
        sys = self.sample(0.1)
      end
      
      dt = sys.dt.not_nil!
      x = x0.dup
      
      t_arr = Array(Float64).new(n_steps)
      x_arr = Array(Float64Tensor).new(n_steps)
      y_arr = Array(Float64Tensor).new(n_steps)
      
      n_steps.times do |i|
        t_arr << i * dt
        x_arr << x
        y = sys.c.matmul(x)
        y_arr << y
        x = sys.a.matmul(x)
      end
      
      {t_arr, x_arr, y_arr}
    end

    # Computes the system bandwidth (the frequency at which magnitude drops by 3dB from DC gain).
    def bandwidth : Float64
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Bandwidth is only supported for SISO systems")
      end

      # Calculate DC Gain magnitude
      dc = dcgain[0, 0].value
      dc_mag = dc.abs
      target_mag = dc_mag / Math.sqrt(2.0) # -3 dB point

      # Sweep frequencies log-spaced
      omega = Float64Tensor.linear_space(-2.0, 4.0, 1000).map { |v| 10.0 ** v }
      h = freqresp(omega)

      best_w = 0.0
      min_diff = Float64::INFINITY
      omega.size.times do |i|
        mag = h.to_unsafe[i].abs
        diff = (mag - target_mag).abs
        if diff < min_diff
          min_diff = diff
          best_w = omega[i].value
        end
      end

      best_w
    end

    # Transforms system to Control Canonical Form.
    def to_control_canonical_form : Tuple(StateSpace, Float64Tensor)
      unless n_inputs == 1
        raise ArgumentError.new("Control canonical form only supported for single-input systems")
      end
      n = n_states
      co = ctrb
      unless is_controllable?
        raise ArgumentError.new("System is not controllable; cannot convert to control canonical form")
      end

      # Find characteristic polynomial coefficients of A: s^n + a1 s^(n-1) + ... + an = 0
      # Using companion matrix properties or Leverrier-Faddeev algorithm
      # Let's use Leverrier-Faddeev algorithm to get polynomial of A:
      poly = Array(Float64).new(n + 1, 0.0)
      poly[0] = 1.0
      curr_a = Float64Tensor.identity(n)
      (1..n).each do |i|
        curr_a = @a.matmul(curr_a)
        trace_val = 0.0
        n.times { |r| trace_val += curr_a[r, r].value }
        ci = -trace_val / i
        poly[i] = ci
        curr_a = curr_a + Float64Tensor.identity(n) * ci
      end

      # Build Transformation Matrix T = Co * M
      # where M is the upper-triangular Toeplitz matrix formed by coefficients [1, a1, a2, ..., a_n-1]
      m_mat = Float64Tensor.zeros([n, n])
      n.times do |i|
        n.times do |j|
          if j >= i
            m_mat[i, j] = poly[j - i]
          end
        end
      end

      t_matrix = co.matmul(m_mat)
      {similarity_transform(t_matrix), t_matrix}
    end

    # Transforms system to Observable Canonical Form.
    def to_observable_canonical_form : Tuple(StateSpace, Float64Tensor)
      unless n_outputs == 1
        raise ArgumentError.new("Observable canonical form only supported for single-output systems")
      end
      n = n_states
      ob = obsv
      unless is_observable?
        raise ArgumentError.new("System is not observable; cannot convert to observable canonical form")
      end

      # Find characteristic polynomial coefficients of A using Leverrier-Faddeev:
      poly = Array(Float64).new(n + 1, 0.0)
      poly[0] = 1.0
      curr_a = Float64Tensor.identity(n)
      (1..n).each do |i|
        curr_a = @a.matmul(curr_a)
        trace_val = 0.0
        n.times { |r| trace_val += curr_a[r, r].value }
        ci = -trace_val / i
        poly[i] = ci
        curr_a = curr_a + Float64Tensor.identity(n) * ci
      end

      # Build Toeplitz matrix M
      m_mat = Float64Tensor.zeros([n, n])
      n.times do |i|
        n.times do |j|
          if j >= i
            m_mat[i, j] = poly[j - i]
          end
        end
      end

      # For observability, transformation is T_inv = M * Ob
      t_inv = m_mat.matmul(ob)
      t_matrix = t_inv.inv
      {similarity_transform(t_matrix), t_matrix}
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
        n.times do |row|
          m.times do |col|
            res.to_unsafe[row * (n * m) + i * m + col] = temp.to_unsafe[row * m + col]
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
        p.times do |row|
          n.times do |col|
            res.to_unsafe[(i * p + row) * n + col] = temp.to_unsafe[row * n + col]
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
        n.times { |i| trace += am.to_unsafe[i * n + i] }
        ak = -trace / index
        den[index] = ak
        m = am + Float64Tensor.identity(n) * ak
      end
      
      num = Array(Float64).new(n + 1, 0.0)
      m = Float64Tensor.identity(n)
      n.times do |k|
        cb = @c.matmul(m).matmul(@b)
        num[k + 1] = cb.to_unsafe[0]
        
        am = @a.matmul(m)
        trace = 0.0
        n.times { |i| trace += am.to_unsafe[i * n + i] }
        ak = -trace / (k + 1)
        m = am + Float64Tensor.identity(n) * ak
      end
      
      d_val = @d.to_unsafe[0]
      (n + 1).times do |i|
        num[i] += d_val * den[i]
      end
      
      while num.size > 1 && num[0].abs < 1e-12
        num.shift
      end

      TransferFunction.new(num.to_tensor, den.to_tensor, @dt)
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

    # State-feedback pole placement using Ackermann's formula.
    # Places poles of (A - B*K) at desired locations.
    # Returns the gain matrix K (1 x n).
    def acker(poles : Array(Float64) | Array(Complex))
      n = n_states
      raise ArgumentError.new("Must specify exactly #{n} poles") if poles.size != n

      a_c = Tensor(Complex, CPU(Complex)).zeros([n, n])
      n.times do |i|
        n.times do |j|
          a_c.to_unsafe[i * n + j] = Complex.new(@a[i, j].value, 0.0)
        end
      end

      phi = Tensor(Complex, CPU(Complex)).eye(n)
      eye_c = Tensor(Complex, CPU(Complex)).eye(n)
      
      poles.each do |p|
        c_pole = p.is_a?(Complex) ? p : Complex.new(p.to_f64, 0.0)
        term = a_c - eye_c * c_pole
        phi = phi.matmul(term)
      end
      
      phi_real = Float64Tensor.zeros([n, n])
      (n * n).times do |i|
        phi_real.to_unsafe[i] = phi.to_unsafe[i].real
      end
      
      co = ctrb
      if !is_controllable?
        raise ArgumentError.new("System is not controllable; cannot place poles")
      end
      
      co_inv = co.inv
      m = n_inputs
      raise NotImplementedError.new("Ackermann's formula only supported for SISO systems") if m != 1
      
      last_row = Float64Tensor.zeros([1, n])
      n.times do |col|
        last_row.to_unsafe[col] = co_inv[n - 1, col].value
      end
      
      last_row.matmul(phi_real)
    end

    # Evaluates system response G(jw) at a set of frequency points.
    # Returns a Tensor of Complex numbers of shape [n_outputs, n_inputs, omega.size].
    def freqresp(omega : Float64Tensor)
      n = n_states
      m = n_inputs
      p = n_outputs
      w_size = omega.size
      
      res = Tensor(Complex, CPU(Complex)).zeros([p, m, w_size])
      
      a_c = Tensor(Complex, CPU(Complex)).zeros([n, n])
      n.times do |i|
        n.times do |j|
          a_c.to_unsafe[i * n + j] = Complex.new(@a.to_unsafe[i * n + j], 0.0)
        end
      end
      
      b_c = Tensor(Complex, CPU(Complex)).zeros([n, m])
      n.times do |i|
        m.times do |j|
          b_c.to_unsafe[i * m + j] = Complex.new(@b.to_unsafe[i * m + j], 0.0)
        end
      end
      
      c_c = Tensor(Complex, CPU(Complex)).zeros([p, n])
      p.times do |i|
        n.times do |j|
          c_c.to_unsafe[i * n + j] = Complex.new(@c.to_unsafe[i * n + j], 0.0)
        end
      end
      
      d_c = Tensor(Complex, CPU(Complex)).zeros([p, m])
      p.times do |i|
        m.times do |j|
          d_c.to_unsafe[i * m + j] = Complex.new(@d.to_unsafe[i * m + j], 0.0)
        end
      end
      
      w_size.times do |idx|
        w = omega.to_unsafe[idx]
        jw_i_minus_a = Tensor(Complex, CPU(Complex)).zeros([n, n])
        n.times do |i|
          n.times do |j|
            val = -a_c.to_unsafe[i * n + j]
            if i == j
              val += Complex.new(0.0, w)
            end
            jw_i_minus_a.to_unsafe[i * n + j] = val
          end
        end
        
        x = jw_i_minus_a.solve(b_c)
        g = c_c.matmul(x) + d_c
        
        p.times do |out_idx|
          m.times do |in_idx|
            res.to_unsafe[(out_idx * m + in_idx) * w_size + idx] = g.to_unsafe[out_idx * m + in_idx]
          end
        end
      end
      
      res
    end

    # Solves the Continuous Algebraic Riccati Equation:
    # A^T * P + P * A - P * B * R^-1 * B^T * P + Q = 0
    # Returns P.
    def care(q : Float64Tensor, r : Float64Tensor)
      n = n_states
      m = n_inputs
      
      g = @b.matmul(r.inv).matmul(@b.transpose)
      
      h = Float64Tensor.zeros([2 * n, 2 * n])
      n.times do |i|
        n.times do |j|
          h.to_unsafe[i * (2 * n) + j] = @a.to_unsafe[i * n + j]
          h.to_unsafe[i * (2 * n) + n + j] = -g.to_unsafe[i * n + j]
          h.to_unsafe[(n + i) * (2 * n) + j] = -q.to_unsafe[i * n + j]
          h.to_unsafe[(n + i) * (2 * n) + n + j] = -@a.to_unsafe[j * n + i]
        end
      end
      
      w, v = h.eig_c
      
      # Reconstruct complex eigenvectors from the real/imag columns returned by LAPACK dgeev
      # Note: v returned from LAPACK is in ColMajor layout
      v_c = Tensor(Complex, CPU(Complex)).zeros([2 * n, 2 * n])
      col = 0
      while col < 2 * n
        is_complex = w.to_unsafe[col].imag.abs > 1e-12
        if is_complex
          (2 * n).times do |row|
            real_val = v.to_unsafe[col * (2 * n) + row]
            imag_val = v.to_unsafe[(col + 1) * (2 * n) + row]
            v_c.to_unsafe[row * (2 * n) + col] = Complex.new(real_val, imag_val)
            v_c.to_unsafe[row * (2 * n) + col + 1] = Complex.new(real_val, -imag_val)
          end
          col += 2
        else
          (2 * n).times do |row|
            v_c.to_unsafe[row * (2 * n) + col] = Complex.new(v.to_unsafe[col * (2 * n) + row], 0.0)
          end
          col += 1
        end
      end
      
      stable_indices = Array(Int32).new
      w.size.times do |i|
        if w.to_unsafe[i].real < 0.0
          stable_indices << i
        end
      end
      
      if stable_indices.size != n
        raise ArgumentError.new("Could not find stable subspace (expected #{n} stable eigenvalues, found #{stable_indices.size})")
      end
      
      u1 = Tensor(Complex, CPU(Complex)).zeros([n, n])
      u2 = Tensor(Complex, CPU(Complex)).zeros([n, n])
      
      n.times do |col_idx|
        orig_col = stable_indices[col_idx]
        n.times do |row_idx|
          u1.to_unsafe[row_idx * n + col_idx] = v_c.to_unsafe[row_idx * (2 * n) + orig_col]
          u2.to_unsafe[row_idx * n + col_idx] = v_c.to_unsafe[(n + row_idx) * (2 * n) + orig_col]
        end
      end
      
      p_complex = u2.matmul(u1.inv)
      
      p_real = Float64Tensor.zeros([n, n])
      (n * n).times do |i|
        p_real.to_unsafe[i] = p_complex.to_unsafe[i].real
      end
      
      p_real
    end

    # Solves continuous-time Linear Quadratic Regulator (LQR) controller: u = -Kx
    # Returns: {K (matrix), P (matrix), closed_loop_poles (Array(Complex))}
    def lqr(q : Float64Tensor, r : Float64Tensor)
      p = care(q, r)
      k = r.inv.matmul(@b.transpose).matmul(p)
      a_cl = @a - @b.matmul(k)
      {k, p, a_cl.eigvals_c.to_a}
    end

    # Solves continuous-time Lyapunov equation: A*P + P*A^T + Q = 0
    # Returns P.
    def lyap(q : Float64Tensor)
      n = n_states
      eye = Float64Tensor.identity(n)
      m_lhs = eye.kron(@a) + @a.kron(eye)
      q_vec = q.reshape([n * n, 1])
      p_vec = m_lhs.solve(-q_vec)
      p_vec.reshape([n, n])
    end

    # Solves discrete-time Lyapunov equation: A*P*A^T - P + Q = 0
    # Returns P.
    def dlyap(q : Float64Tensor)
      n = n_states
      eye_nn = Float64Tensor.identity(n * n)
      m_lhs = @a.kron(@a) - eye_nn
      q_vec = q.reshape([n * n, 1])
      p_vec = m_lhs.solve(-q_vec)
      p_vec.reshape([n, n])
    end

    # Computes classical gain and phase margins (Bode margins) for a SISO system.
    # Returns: {GM (amplitude gain margin), GM_dB (gain margin in dB), PM (phase margin in degrees), w_gc (gain crossover freq), w_pc (phase crossover freq)}
    def stability_margins
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Stability margins only supported for SISO systems")
      end

      omega = Float64Tensor.linear_space(0.01, 1000.0, 10000)
      h = freqresp(omega)
      
      w_gc = -1.0
      w_pc = -1.0
      
      pm = 0.0
      gm = Float64::INFINITY
      
      wrap_phase = ->(p : Float64) {
        val = p % (2.0 * Math::PI)
        val -= 2.0 * Math::PI if val > Math::PI
        val += 2.0 * Math::PI if val < -Math::PI
        val
      }

      n_points = omega.size
      mags = Array(Float64).new(n_points)
      phases = Array(Float64).new(n_points)
      
      n_points.times do |i|
        val = h.to_unsafe[i]
        mags << val.abs
        phases << wrap_phase.call(Math.atan2(val.imag, val.real))
      end
      
      (n_points - 1).times do |i|
        if (mags[i] - 1.0) * (mags[i+1] - 1.0) <= 0.0
          t = (1.0 - mags[i]) / (mags[i+1] - mags[i])
          w = omega[i].value + t * (omega[i+1].value - omega[i].value)
          w_gc = w
          interpolated_phase = phases[i] + t * (phases[i+1] - phases[i])
          pm = wrap_phase.call(interpolated_phase + Math::PI) * 180.0 / Math::PI
          break
        end
      end
      
      (n_points - 1).times do |i|
        p1 = phases[i]
        p2 = phases[i+1]
        
        if p1 * p2 < 0.0 && p1.abs > 2.0 && p2.abs > 2.0
          t = (-Math::PI - p1) / (p2 - p1) rescue 0.5
          w = omega[i].value + t * (omega[i+1].value - omega[i].value)
          w_pc = w
          interpolated_mag = mags[i] + t * (mags[i+1] - mags[i])
          gm = 1.0 / interpolated_mag if interpolated_mag > 0.0
          break
        end
      end
      
      gm_db = gm == Float64::INFINITY ? Float64::INFINITY : 20.0 * Math.log10(gm)
      
      {gm, gm_db, pm, w_gc, w_pc}
    end

    # Solves the Discrete-Time Algebraic Riccati Equation (DARE) using iterative method:
    # A^T * P * A - P - A^T * P * B * (R + B^T * P * B)^-1 * B^T * P * A + Q = 0
    # Returns P.
    def dare(q : Float64Tensor, r : Float64Tensor, max_iter = 1000, tol = 1e-9)
      n = n_states
      p = q.dup
      
      a_t = @a.transpose
      b_t = @b.transpose
      
      max_iter.times do
        # temp = R + B^T * P * B
        temp = r + b_t.matmul(p).matmul(@b)
        
        # P_next = A^T * P * A - A^T * P * B * inv(R + B^T * P * B) * B^T * P * A + Q
        term1 = a_t.matmul(p).matmul(@a)
        term2 = a_t.matmul(p).matmul(@b).matmul(temp.inv).matmul(b_t).matmul(p).matmul(@a)
        p_next = term1 - term2 + q
        
        # Check convergence
        diff = 0.0
        (n * n).times do |i|
          d_val = (p_next.to_unsafe[i] - p.to_unsafe[i]).abs
          diff = d_val if d_val > diff
        end
        
        p = p_next
        break if diff < tol
      end
      p
    end

    # Solves discrete-time Linear Quadratic Regulator (DLQR) controller: u = -Kx
    # Returns: {K (matrix), P (matrix), closed_loop_poles (Array(Complex))}
    def dlqr(q : Float64Tensor, r : Float64Tensor)
      p = dare(q, r)
      b_t = @b.transpose
      temp = r + b_t.matmul(p).matmul(@b)
      k = temp.inv.matmul(b_t).matmul(p).matmul(@a)
      a_cl = @a - @b.matmul(k)
      {k, p, a_cl.eigvals_c.to_a}
    end

    # Computes estimator gain L for a full-state observer via duality:
    # error dynamics error_k+1 = (A - L*C)*error_k
    # Returns the observer gain L (n x 1).
    def acker_obs(poles : Array(Float64) | Array(Complex))
      # Duality: closed-loop observer dynamics (A - L*C) is dual to controller (A^T - C^T*L^T)
      sys_dual = StateSpace.new(@a.transpose, @c.transpose, @b.transpose, @d.transpose)
      k_dual = sys_dual.acker(poles)
      k_dual.transpose
    end

    # Computes Controllability (:c) or Observability (:o) Gramian.
    # Returns Gramian matrix P.
    def gram(type : Symbol)
      if type == :c
        # Controllability Gramian: A*Wc + Wc*A^T + B*B^T = 0
        q = @b.matmul(@b.transpose)
        lyap(q)
      elsif type == :o
        # Observability Gramian: A^T*Wo + Wo*A + C^T*C = 0
        q = @c.transpose.matmul(@c)
        sys_t = StateSpace.new(@a.transpose, @b, @c, @d)
        sys_t.lyap(q)
      else
        raise ArgumentError.new("Gramian type must be :c (controllability) or :o (observability)")
      end
    end

    # Computes the Hankel Singular Values of the system.
    # Returns a sorted array of Singular Values.
    def hsvd
      wc = gram(:c)
      wo = gram(:o)
      prod = wc.matmul(wo)
      
      # For a positive-semidefinite product, eigenvalues are real and non-negative
      eigvals = prod.eigvals
      res = Array(Float64).new(eigvals.size)
      eigvals.size.times do |i|
        res << Math.sqrt(eigvals.to_unsafe[i].abs)
      end
      res.sort.reverse
    end

    # Simulates time response with arbitrary input (wrapper for simulate).
    def lsim(u : Float64Tensor, t : Float64Tensor, x0 : Float64Tensor? = nil, method = :rk4)
      simulate(t, x0: x0, u: u, method: method)
    end

    # Converts a SISO StateSpace system to Observability Canonical Form.
    def to_observability_form
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Observability form only supported for SISO systems")
      end
      tf = to_transferfunction
      n = tf.den.size - 1
      if n == 0
        return StateSpace.new(
          Float64Tensor.zeros([0, 0]),
          Float64Tensor.zeros([0, 1]),
          Float64Tensor.zeros([1, 0]),
          [[tf.num[0].value]].to_tensor,
          @dt
        )
      end
      
      ss_c = tf.to_statespace
      StateSpace.new(ss_c.a.transpose, ss_c.c.transpose, ss_c.b.transpose, ss_c.d, @dt)
    end

    # Converts a StateSpace system to Modal (Diagonal) Form.
    def to_modal_form
      n = n_states
      w, v = @a.eig_c
      
      # Form transformation matrix T using real and imaginary parts of eigenvectors
      t_matrix = Float64Tensor.zeros([n, n])
      col = 0
      while col < n
        is_complex = w.to_unsafe[col].imag.abs > 1e-9
        if is_complex
          n.times do |row|
            t_matrix.to_unsafe[row * n + col] = v.to_unsafe[col * n + row]
            t_matrix.to_unsafe[row * n + col + 1] = v.to_unsafe[(col + 1) * n + row]
          end
          col += 2
        else
          n.times do |row|
            t_matrix.to_unsafe[row * n + col] = v.to_unsafe[col * n + row]
          end
          col += 1
        end
      end
      
      t_inv = t_matrix.inv
      a_m = t_inv.matmul(@a).matmul(t_matrix)
      b_m = t_inv.matmul(@b)
      c_m = @c.matmul(t_matrix)
      StateSpace.new(a_m, b_m, c_m, @d, @dt)
    end

    # Evaluates closed-loop poles as gain K varies from 0 to infinity for a SISO system.
    # Returns an array of pole arrays.
    def root_locus(gains : Float64Tensor)
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Root locus only supported for SISO systems")
      end
      
      n = n_states
      m_gains = gains.size
      res = Array(Array(Complex)).new(m_gains)
      
      m_gains.times do |i|
        k = gains.to_unsafe[i]
        a_cl = @a - @b.matmul(@c) * k
        res << a_cl.eigvals_c.to_a
      end
      res
    end

    # Returns Bode data: {omega, magnitudes (dB), phases (degrees)} for a SISO system.
    def bode_data(omega : Float64Tensor)
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Bode data only supported for SISO systems")
      end

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

    # Returns Nyquist data: {real_parts, imaginary_parts} for a SISO system.
    def nyquist_data(omega : Float64Tensor)
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Nyquist data only supported for SISO systems")
      end

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

    # Helper method to compute matrix logarithm using eigenvalue decomposition.
    def self.logm(matrix : Float64Tensor) : Tensor(Complex, CPU(Complex))
      n = matrix.shape[0]
      w, v = matrix.eig_c
      
      d_log = Tensor(Complex, CPU(Complex)).zeros([n, n])
      n.times do |i|
        val = w.to_unsafe[i]
        r = val.abs
        theta = Math.atan2(val.imag, val.real)
        d_log.to_unsafe[i * n + i] = Complex.new(Math.log(r), theta)
      end
      
      v_c = Tensor(Complex, CPU(Complex)).zeros([n, n])
      col = 0
      while col < n
        is_complex = w.to_unsafe[col].imag.abs > 1e-12
        if is_complex
          n.times do |row|
            real_val = v.to_unsafe[col * n + row]
            imag_val = v.to_unsafe[(col + 1) * n + row]
            v_c.to_unsafe[row * n + col] = Complex.new(real_val, imag_val)
            v_c.to_unsafe[row * n + col + 1] = Complex.new(real_val, -imag_val)
          end
          col += 2
        else
          n.times do |row|
            v_c.to_unsafe[row * n + col] = Complex.new(v.to_unsafe[col * n + row], 0.0)
          end
          col += 1
        end
      end
      v_c.matmul(d_log).matmul(v_c.inv)
    end

    # Converts a discrete-time StateSpace system to continuous-time (D2C).
    def to_continuous
      dt = @dt
      if dt.nil? || dt <= 0
        raise "System is already continuous"
      end
      
      n = n_states
      m = n_inputs
      
      # Form block matrix M = [Ad Bd; 0 I]
      block = Float64Tensor.zeros([n + m, n + m])
      n.times do |r|
        n.times do |c|
          block[r, c] = @a[r, c].value
        end
        m.times do |c|
          block[r, n + c] = @b[r, c].value
        end
      end
      m.times do |r|
        block[n + r, n + r] = 1.0
      end
      
      log_m = StateSpace.logm(block)
      
      ac = Float64Tensor.zeros([n, n])
      bc = Float64Tensor.zeros([n, m])
      
      n.times do |r|
        n.times do |c|
          ac[r, c] = log_m.to_unsafe[r * (n + m) + c].real / dt
        end
        m.times do |c|
          bc[r, c] = log_m.to_unsafe[r * (n + m) + n + c].real / dt
        end
      end
      
      StateSpace.new(ac, bc, @c, @d, nil)
    end

    # Computes a balanced truncation of the system.
    # Returns a reduced-order StateSpace system of size `orders`.
    def balred(orders : Int32)
      if orders > n_states
        raise ArgumentError.new("Reduced order must be less than or equal to current order")
      end
      if orders <= 0
        raise ArgumentError.new("Reduced order must be greater than 0")
      end
      
      wc = gram(:c)
      wo = gram(:o)
      
      # Step 2: Compute SVD of Gramians to find Cholesky-like factors
      uc, sc, vct = wc.svd
      uo, so, vot = wo.svd
      
      n = n_states
      lc = Float64Tensor.zeros([n, n])
      lo = Float64Tensor.zeros([n, n])
      
      n.times do |i|
        s_c_val = Math.sqrt(sc[i].value.abs)
        s_o_val = Math.sqrt(so[i].value.abs)
        
        n.times do |j|
          lc[j, i] = uc[j, i] * s_c_val
          lo[j, i] = uo[j, i] * s_o_val
        end
      end
      
      # Step 3: SVD of Lo^T * Lc
      product = lo.transpose.matmul(lc)
      u_p, s_p, vt_p = product.svd
      v_p = vt_p.transpose
      
      # Step 4: Construct transformation matrices T and T_inv
      t_matrix = Float64Tensor.zeros([n, n])
      t_inv = Float64Tensor.zeros([n, n])
      
      n.times do |i|
        sig_val = s_p[i].value
        sig_factor = sig_val.abs > 1e-12 ? 1.0 / Math.sqrt(sig_val) : 0.0
        
        # Column i of T = Lc * col_i(V) * sig_factor
        n.times do |r|
          sum = 0.0
          n.times do |k|
            sum += lc[r, k].value * v_p[k, i].value
          end
          t_matrix[r, i] = sum * sig_factor
        end
        
        # Row i of T_inv = sig_factor * col_i(U)^T * Lo^T
        n.times do |c_idx|
          sum = 0.0
          n.times do |k|
            sum += u_p[k, i].value * lo[c_idx, k].value
          end
          t_inv[i, c_idx] = sum * sig_factor
        end
      end
      
      # Transform full system to balanced realization
      ab = t_inv.matmul(@a).matmul(t_matrix)
      bb = t_inv.matmul(@b)
      cb = @c.matmul(t_matrix)
      
      # Step 5: Truncate to first `orders` states
      ar = Float64Tensor.zeros([orders, orders])
      br = Float64Tensor.zeros([orders, n_inputs])
      cr = Float64Tensor.zeros([n_outputs, orders])
      
      orders.times do |r|
        orders.times do |c_idx|
          ar[r, c_idx] = ab[r, c_idx].value
        end
        n_inputs.times do |c_idx|
          br[r, c_idx] = bb[r, c_idx].value
        end
      end
      n_outputs.times do |r|
        orders.times do |c_idx|
          cr[r, c_idx] = cb[r, c_idx].value
        end
      end
      
      StateSpace.new(ar, br, cr, @d, @dt)
    end

    # Designs a Linear Quadratic Gaussian (LQG) controller.
    # Returns a StateSpace controller that takes output measurements y
    # as input and produces control signals u.
    def lqg(k_gain : Float64Tensor, l_gain : Float64Tensor) : StateSpace
      n = n_states
      m = n_inputs
      p = n_outputs
      
      bk = @b.matmul(k_gain)
      lc = l_gain.matmul(@c)
      ldk = l_gain.matmul(@d).matmul(k_gain)
      
      a_lqg = @a - bk - lc + ldk
      b_lqg = l_gain
      c_lqg = k_gain
      d_lqg = Float64Tensor.zeros([m, p])
      
      StateSpace.new(a_lqg, b_lqg, c_lqg, d_lqg, @dt)
    end

    # Returns Nichols data: {omega, magnitudes (dB), phases (degrees)} for a SISO system.
    def nichols_data(omega : Float64Tensor)
      _, db, deg = bode_data(omega)
      {omega, db, deg}
    end

    # Performs coordinate transformation z = T * x (i.e. x = T_inv * z)
    def similarity_transform(t_matrix : Float64Tensor) : StateSpace
      t_inv = t_matrix.inv
      a_new = t_inv.matmul(@a).matmul(t_matrix)
      b_new = t_inv.matmul(@b)
      c_new = @c.matmul(t_matrix)
      StateSpace.new(a_new, b_new, c_new, @d, @dt)
    end

    # Computes the controllability decomposition (controllable / uncontrollable parts).
    # Returns {transformed_system, T_matrix, controllable_rank}
    def controllable_decomposition : Tuple(StateSpace, Float64Tensor, Int32)
      n = n_states
      m = n_inputs
      
      co = Float64Tensor.zeros([n, n * m])
      curr = @b.dup
      n.times do |i|
        m.times do |c_idx|
          n.times do |r_idx|
            co[r_idx, i * m + c_idx] = curr[r_idx, c_idx].value
          end
        end
        curr = @a.matmul(curr) if i < n - 1
      end
      
      u, s, vt = co.svd
      
      r = 0
      s.size.times do |i|
        r += 1 if s[i].value > 1e-9
      end
      
      t_matrix = u
      t_inv = t_matrix.transpose
      
      a_c = t_inv.matmul(@a).matmul(t_matrix)
      b_c = t_inv.matmul(@b)
      c_c = @c.matmul(t_matrix)
      
      {StateSpace.new(a_c, b_c, c_c, @d, @dt), t_matrix, r}
    end

    # Computes the observability decomposition (observable / unobservable parts).
    # Returns {transformed_system, T_matrix, observable_rank}
    def observable_decomposition : Tuple(StateSpace, Float64Tensor, Int32)
      n = n_states
      p = n_outputs
      
      ob = Float64Tensor.zeros([n * p, n])
      curr = @c.dup
      n.times do |i|
        p.times do |r_idx|
          n.times do |c_idx|
            ob[i * p + r_idx, c_idx] = curr[r_idx, c_idx].value
          end
        end
        curr = curr.matmul(@a) if i < n - 1
      end
      
      u, s, vt = ob.svd
      
      r = 0
      s.size.times do |i|
        r += 1 if s[i].value > 1e-9
      end
      
      t_matrix = vt.transpose
      t_inv = vt
      
      a_o = t_inv.matmul(@a).matmul(t_matrix)
      b_o = t_inv.matmul(@b)
      c_o = @c.matmul(t_matrix)
      
      {StateSpace.new(a_o, b_o, c_o, @d, @dt), t_matrix, r}
    end

    # Computes the minimal realization of the system.
    # Eliminates uncontrollable and unobservable states.
    def minreal : StateSpace
      n = n_states
      return self if n == 0
      
      # Step 1: Controllability Decomposition
      sys_c, _, r_c = controllable_decomposition
      return StateSpace.new(Float64Tensor.zeros([0, 0]), Float64Tensor.zeros([0, n_inputs]), Float64Tensor.zeros([n_outputs, 0]), @d, @dt) if r_c == 0
      
      a_c = Float64Tensor.zeros([r_c, r_c])
      b_c = Float64Tensor.zeros([r_c, n_inputs])
      c_c = Float64Tensor.zeros([n_outputs, r_c])
      
      r_c.times do |r|
        r_c.times do |c_idx|
          a_c[r, c_idx] = sys_c.a[r, c_idx].value
        end
        n_inputs.times do |c_idx|
          b_c[r, c_idx] = sys_c.b[r, c_idx].value
        end
      end
      n_outputs.times do |r|
        r_c.times do |c_idx|
          c_c[r, c_idx] = sys_c.c[r, c_idx].value
        end
      end
      
      sys_controllable = StateSpace.new(a_c, b_c, c_c, @d, @dt)
      
      # Step 2: Observability Decomposition on the controllable subsystem
      sys_o, _, r_o = sys_controllable.observable_decomposition
      return StateSpace.new(Float64Tensor.zeros([0, 0]), Float64Tensor.zeros([0, n_inputs]), Float64Tensor.zeros([n_outputs, 0]), @d, @dt) if r_o == 0
      
      a_o = Float64Tensor.zeros([r_o, r_o])
      b_o = Float64Tensor.zeros([r_o, n_inputs])
      c_o = Float64Tensor.zeros([n_outputs, r_o])
      
      r_o.times do |r|
        r_o.times do |c_idx|
          a_o[r, c_idx] = sys_o.a[r, c_idx].value
        end
        n_inputs.times do |c_idx|
          b_o[r, c_idx] = sys_o.b[r, c_idx].value
        end
      end
      n_outputs.times do |r|
        r_o.times do |c_idx|
          c_o[r, c_idx] = sys_o.c[r, c_idx].value
        end
      end
      
      StateSpace.new(a_o, b_o, c_o, @d, @dt)
    end

    # Augments the state-space system with an integrator for tracking control.
    # Returns the augmented StateSpace system.
    def augment_integrator : StateSpace
      n = n_states
      m = n_inputs
      p = n_outputs
      
      a_aug = Float64Tensor.zeros([n + p, n + p])
      n.times do |r|
        n.times do |c_idx|
          a_aug[r, c_idx] = @a[r, c_idx].value
        end
      end
      p.times do |r|
        n.times do |c_idx|
          a_aug[n + r, c_idx] = -@c[r, c_idx].value
        end
      end
      
      b_aug = Float64Tensor.zeros([n + p, m])
      n.times do |r|
        m.times do |c_idx|
          b_aug[r, c_idx] = @b[r, c_idx].value
        end
      end
      p.times do |r|
        m.times do |c_idx|
          b_aug[n + r, c_idx] = -@d[r, c_idx].value
        end
      end
      
      c_aug = Float64Tensor.zeros([p, n + p])
      p.times do |r|
        n.times do |c_idx|
          c_aug[r, c_idx] = @c[r, c_idx].value
        end
      end
      
      d_aug = Float64Tensor.zeros([p, m])
      p.times do |r|
        m.times do |c_idx|
          d_aug[r, c_idx] = @d[r, c_idx].value
        end
      end
      
      StateSpace.new(a_aug, b_aug, c_aug, d_aug, @dt)
    end

    # Computes the optimal Kalman estimator gain L for continuous systems.
    # Q is process noise covariance matrix, R is measurement noise covariance matrix.
    def lqe(q_noise : Float64Tensor, r_noise : Float64Tensor) : Float64Tensor
      dual_sys = StateSpace.new(@a.transpose, @c.transpose, @b.transpose, @d.transpose, @dt)
      p = dual_sys.care(q_noise, r_noise)
      p.matmul(@c.transpose).matmul(r_noise.inv)
    end

    # Computes the optimal Kalman estimator gain L for discrete systems.
    # Q is process noise covariance matrix, R is measurement noise covariance matrix.
    def dlqe(q_noise : Float64Tensor, r_noise : Float64Tensor) : Float64Tensor
      dual_sys = StateSpace.new(@a.transpose, @c.transpose, @b.transpose, @d.transpose, @dt)
      p = dual_sys.dare(q_noise, r_noise)
      c_p_ct = @c.matmul(p).matmul(@c.transpose)
      inv_term = (c_p_ct + r_noise).inv
      @a.matmul(p).matmul(@c.transpose).matmul(inv_term)
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
