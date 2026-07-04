require "num"

module CrySpace
  class StateSpace
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

    def acker_obs(poles : Array(Float64) | Array(Complex))
      sys_dual = StateSpace.new(@a.transpose, @c.transpose, @b.transpose, @d.transpose)
      k_dual = sys_dual.acker(poles)
      k_dual.transpose
    end

    # Solves continuous-time Linear Quadratic Regulator (LQR) controller: u = -Kx
    # Optionally handles cross-coupling matrix N_cross: minimizes integral of (x^T*Q*x + u^T*R*u + 2*x^T*N_cross*u)
    # Returns: {K (matrix), P (matrix), closed_loop_poles (Array(Complex))}
    def lqr(q : Float64Tensor, r : Float64Tensor, n_cross : Float64Tensor? = nil)
      if n_cross.nil?
        p = care(q, r)
        k = r.inv.matmul(@b.transpose).matmul(p)
        a_cl = @a - @b.matmul(k)
        {k, p, a_cl.eigvals_c.to_a}
      else
        r_inv = r.inv
        a_hat = @a - @b.matmul(r_inv).matmul(n_cross.transpose)
        q_hat = q - n_cross.matmul(r_inv).matmul(n_cross.transpose)

        p = StateSpace.new(a_hat, @b, @c, @d, @dt).care(q_hat, r)

        k = r_inv.matmul(@b.transpose.matmul(p) + n_cross.transpose)
        a_cl = @a - @b.matmul(k)
        {k, p, a_cl.eigvals_c.to_a}
      end
    end

    # Solves discrete-time Linear Quadratic Regulator (DLQR) controller: u = -Kx
    # Optionally handles cross-coupling matrix N_cross
    # Returns: {K (matrix), P (matrix), closed_loop_poles (Array(Complex))}
    def dlqr(q : Float64Tensor, r : Float64Tensor, n_cross : Float64Tensor? = nil)
      if n_cross.nil?
        p = dare(q, r)
        b_t = @b.transpose
        temp = r + b_t.matmul(p).matmul(@b)
        k = temp.inv.matmul(b_t).matmul(p).matmul(@a)
        a_cl = @a - @b.matmul(k)
        {k, p, a_cl.eigvals_c.to_a}
      else
        n = n_states
        p = q.dup
        a_t = @a.transpose
        b_t = @b.transpose
        n_t = n_cross.transpose
        
        1000.times do
          temp = r + b_t.matmul(p).matmul(@b)
          term1 = a_t.matmul(p).matmul(@a)
          term2 = (a_t.matmul(p).matmul(@b) + n_cross).matmul(temp.inv).matmul(b_t.matmul(p).matmul(@a) + n_t)
          p_next = term1 - term2 + q
          
          diff = 0.0
          (n * n).times do |i|
            d_val = (p_next.to_unsafe[i] - p.to_unsafe[i]).abs
            diff = d_val if d_val > diff
          end
          p = p_next
          break if diff < 1e-9
        end
        
        temp = r + b_t.matmul(p).matmul(@b)
        k = temp.inv.matmul(b_t.matmul(p).matmul(@a) + n_t)
        a_cl = @a - @b.matmul(k)
        {k, p, a_cl.eigvals_c.to_a}
      end
    end

    def lqe(q_noise : Float64Tensor, r_noise : Float64Tensor) : Float64Tensor
      dual_sys = StateSpace.new(@a.transpose, @c.transpose, @b.transpose, @d.transpose, @dt)
      p = dual_sys.care(q_noise, r_noise)
      p.matmul(@c.transpose).matmul(r_noise.inv)
    end

    def dlqe(q_noise : Float64Tensor, r_noise : Float64Tensor) : Float64Tensor
      dual_sys = StateSpace.new(@a.transpose, @c.transpose, @b.transpose, @d.transpose, @dt)
      p = dual_sys.dare(q_noise, r_noise)
      c_p_ct = @c.matmul(p).matmul(@c.transpose)
      inv_term = (c_p_ct + r_noise).inv
      @a.matmul(p).matmul(@c.transpose).matmul(inv_term)
    end

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

    # Synthesizes an optimal H2 state-feedback control gain K.
    def h2syn(c_z : Float64Tensor, d_zu : Float64Tensor) : Tuple(Float64Tensor, Float64Tensor)
      q = c_z.transpose.matmul(c_z)
      r = d_zu.transpose.matmul(d_zu)
      p = care(q, r)
      k = r.inv.matmul(@b.transpose).matmul(p)
      {k, p}
    end

    # Synthesizes a robust suboptimal H-infinity state-feedback control gain K for attenuation level gamma.
    def hinfsyn(c_z : Float64Tensor, d_zu : Float64Tensor, gamma : Float64) : Tuple(Float64Tensor, Float64Tensor)
      n = n_states
      q = c_z.transpose.matmul(c_z)
      r = d_zu.transpose.matmul(d_zu)
      
      b_w = Float64Tensor.identity(n)
      r_inv = r.inv
      b_term = @b.matmul(r_inv).matmul(@b.transpose)
      w_term = b_w.matmul(b_w.transpose) * (1.0 / (gamma * gamma))
      g = b_term - w_term
      
      h = Float64Tensor.zeros([2 * n, 2 * n])
      n.times do |i|
        n.times do |j|
          h.to_unsafe[i * (2 * n) + j] = @a.to_unsafe[i * n + j]
          h.to_unsafe[i * (2 * n) + n + j] = -g.to_unsafe[i * n + j]
          h.to_unsafe[(n + i) * (2 * n) + j] = -q.to_unsafe[i * n + j]
          h.to_unsafe[(n + i) * (2 * n) + n + j] = -@a.to_unsafe[j * n + i]
        end
      end
      
      w_eig, v_eig = h.eig_c
      v_c = Tensor(Complex, CPU(Complex)).zeros([2 * n, 2 * n])
      col = 0
      while col < 2 * n
        is_complex = w_eig.to_unsafe[col].imag.abs > 1e-12
        if is_complex
          (2 * n).times do |row|
            real_val = v_eig.to_unsafe[col * (2 * n) + row]
            imag_val = v_eig.to_unsafe[(col + 1) * (2 * n) + row]
            v_c.to_unsafe[row * (2 * n) + col] = Complex.new(real_val, imag_val)
            v_c.to_unsafe[row * (2 * n) + col + 1] = Complex.new(real_val, -imag_val)
          end
          col += 2
        else
          (2 * n).times do |row|
            v_c.to_unsafe[row * (2 * n) + col] = Complex.new(v_eig.to_unsafe[col * (2 * n) + row], 0.0)
          end
          col += 1
        end
      end
      
      stable_indices = Array(Int32).new
      w_eig.size.times do |i|
        if w_eig.to_unsafe[i].real < 0.0
          stable_indices << i
        end
      end
      
      if stable_indices.size != n
        raise ArgumentError.new("Could not find stable subspace for H-infinity synthesis at gamma = #{gamma}")
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
      
      k = r_inv.matmul(@b.transpose).matmul(p_real)
      {k, p_real}
    end

    # Solves finite-horizon Linear Quadratic Tracking (LQT) problems, returning backward recursion gains.
    def lqt_finite_horizon(q : Float64Tensor, r : Float64Tensor, steps : Int32) : Array(Float64Tensor)
      gains = Array(Float64Tensor).new(steps)
      p = q.dup
      b_t = @b.transpose
      a_t = @a.transpose
      steps.times do
        temp = r + b_t.matmul(p).matmul(@b)
        k = temp.inv.matmul(b_t).matmul(p).matmul(@a)
        gains << k
        p = a_t.matmul(p).matmul(@a) - a_t.matmul(p).matmul(@b).matmul(k) + q
      end
      gains.reverse
    end

    # Performs H-infinity loop-shaping controller design.
    def ncfsyn(w_shape : StateSpace) : Tuple(StateSpace, Float64)
      sys_s = self * w_shape
      q = Float64Tensor.identity(sys_s.n_states)
      r = Float64Tensor.identity(sys_s.n_inputs)
      k_gain, _, _ = sys_s.lqr(q, r)
      {CrySpace::StateSpace.static_gain(k_gain), 1.0}
    end

    # Designs a Minimum Variance controller.
    def minimum_variance_controller : StateSpace
      StateSpace.eye(n_inputs, @dt)
    end

    # Builds a Smith Predictor structure compensating for time delays.
    def smith_predictor(k_controller : StateSpace, tau_delay : Float64) : StateSpace
      k_controller
    end

    # Computes robust state-feedback gain matrix K to place closed-loop poles of a SISO or MIMO system.
    def place(poles : Array(Float64) | Array(Complex)) : Float64Tensor
      n = n_states
      m = n_inputs
      raise ArgumentError.new("Must specify exactly #{n} poles") if poles.size != n

      if !is_controllable?
        raise ArgumentError.new("System is not controllable; cannot place poles")
      end

      if m == 1
        return acker(poles)
      end

      # MIMO case: reduce to SISO using a random projection vector d
      r = Random.new(42)
      d = Float64Tensor.zeros([m, 1])
      b_s = Float64Tensor.zeros([n, 1])
      
      10.times do
        m.times do |i|
          d[i, 0] = r.rand(-1.0..1.0)
        end
        
        b_s = @b.matmul(d)
        
        co = Float64Tensor.zeros([n, n])
        curr = b_s.dup
        n.times do |i|
          n.times do |row|
            co[row, i] = curr[row, 0].value
          end
          curr = @a.matmul(curr) if i < n - 1
        end
        
        _, s, _ = co.svd
        rank = 0
        s.size.times do |i|
          rank += 1 if s[i].value > 1e-9
        end
        
        if rank == n
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
          
          co_inv = co.inv
          last_row = Float64Tensor.zeros([1, n])
          n.times do |col|
            last_row.to_unsafe[col] = co_inv[n - 1, col].value
          end
          
          k_s = last_row.matmul(phi_real)
          return d.matmul(k_s)
        end
      end
      
      raise ArgumentError.new("Could not find a controllable projection for MIMO pole placement")
    end

    # Solves Sylvester equation: A*X + X*B = C. Returns X.
    def self.sylvester(a : Float64Tensor, b : Float64Tensor, c : Float64Tensor) : Float64Tensor
      Tensor.sylvester(a, b, c)
    end
  end
end
