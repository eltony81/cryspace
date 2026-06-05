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
      
      ad = (@a * dt).expm
      
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
          a_c.to_unsafe[i * n + j] = Complex.new(@a.to_unsafe[i * n + j], 0.0)
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
        last_row.to_unsafe[col] = co_inv.to_unsafe[(n - 1) * n + col]
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
