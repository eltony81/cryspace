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
    # Returns: {K (matrix), P (matrix), closed_loop_poles (Array(Complex))}
    def lqr(q : Float64Tensor, r : Float64Tensor)
      p = care(q, r)
      k = r.inv.matmul(@b.transpose).matmul(p)
      a_cl = @a - @b.matmul(k)
      {k, p, a_cl.eigvals_c.to_a}
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
  end
end
