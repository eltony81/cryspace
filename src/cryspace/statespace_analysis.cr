require "num"

module CrySpace
  class StateSpace
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

    # Solves the Discrete-Time Algebraic Riccati Equation (DARE) using iterative method:
    # A^T * P * A - P - A^T * P * B * (R + B^T * P * B)^-1 * B^T * P * A + Q = 0
    # Returns P.
    def dare(q : Float64Tensor, r : Float64Tensor, max_iter = 1000, tol = 1e-9)
      n = n_states
      p = q.dup
      
      a_t = @a.transpose
      b_t = @b.transpose
      
      max_iter.times do
        temp = r + b_t.matmul(p).matmul(@b)
        term1 = a_t.matmul(p).matmul(@a)
        term2 = a_t.matmul(p).matmul(@b).matmul(temp.inv).matmul(b_t).matmul(p).matmul(@a)
        p_next = term1 - term2 + q
        
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

    def nichols_data(omega : Float64Tensor)
      _, db, deg = bode_data(omega)
      {omega, db, deg}
    end

    # Computes classical gain and phase margins (Bode margins) for a SISO system.
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

    # Computes the system bandwidth (the frequency at which magnitude drops by 3dB from DC gain).
    def bandwidth : Float64
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Bandwidth is only supported for SISO systems")
      end

      dc = dcgain[0, 0].value
      dc_mag = dc.abs
      target_mag = dc_mag / Math.sqrt(2.0)

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

    # Computes the Peak Gain (H-infinity norm) of a SISO system.
    def peak_gain : Float64
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Peak gain is only supported for SISO systems")
      end
      omega = Float64Tensor.linear_space(-3.0, 5.0, 2000).map { |v| 10.0 ** v }
      h = freqresp(omega)
      
      max_mag = 0.0
      h.size.times do |i|
        mag = h.to_unsafe[i].abs
        max_mag = mag if mag > max_mag
      end
      max_mag
    end

    # Computes the multi-loop/disk margin (or classical loop margins using Nyquist analysis).
    def loop_margins : Tuple(Tuple(Float64, Float64), Tuple(Float64, Float64))
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("Loop margins are only supported for SISO systems")
      end
      
      omega = Float64Tensor.linear_space(-2.0, 4.0, 1000).map { |v| 10.0 ** v }
      h = freqresp(omega)

      min_dist = Float64::INFINITY
      h.size.times do |i|
        g_val = h.to_unsafe[i]
        dist = (g_val + 1.0).abs
        min_dist = dist if dist < min_dist
      end

      min_dist = 1e-4 if min_dist < 1e-4
      gm_low = 1.0 / (1.0 + min_dist)
      gm_high = min_dist >= 1.0 ? Float64::INFINITY : 1.0 / (1.0 - min_dist)
      pm_rad = 2.0 * Math.asin({min_dist / 2.0, 1.0}.min)
      pm_deg = pm_rad * 180.0 / Math::PI

      { {gm_low, gm_high}, {-pm_deg, pm_deg} }
    end

    # Computes singular values of the frequency response matrix G(jw) over a range of frequency points.
    # Returns: {omega, singular_values_matrix [omega.size, min(outputs, inputs)]}
    def sigma_data(omega : Float64Tensor)
      h = freqresp(omega)
      n_out = n_outputs
      n_in = n_inputs
      w_size = omega.size
      n_sv = {n_out, n_in}.min
      
      res = Float64Tensor.zeros([w_size, n_sv])
      
      w_size.times do |idx|
        m_real = Float64Tensor.zeros([2 * n_out, 2 * n_in])
        n_out.times do |r|
          n_in.times do |c|
            val = h.to_unsafe[(r * n_in + c) * w_size + idx]
            real_val = val.real
            imag_val = val.imag
            
            m_real[r, c] = real_val
            m_real[r, n_in + c] = -imag_val
            m_real[n_out + r, c] = imag_val
            m_real[n_out + r, n_in + c] = real_val
          end
        end
        
        _, s, _ = m_real.svd
        n_sv.times do |i|
          res[idx, i] = s[2 * i].value
        end
      end
      
      {omega, res}
    end

    # Computes prefilter tracking scaling gain N for zero steady-state tracking error under LQR control u = -Kx + Nr
    def nbar(k_gain : Float64Tensor) : Float64Tensor
      n = n_states
      if @dt.nil? || @dt == 0
        a_cl = @a - @b.matmul(k_gain)
        inv_acl = a_cl.inv
        term = @c.matmul(inv_acl).matmul(@b)
        (-term + @d).inv
      else
        eye = Float64Tensor.identity(n)
        a_cl = eye - (@a - @b.matmul(k_gain))
        inv_acl = a_cl.inv
        term = @c.matmul(inv_acl).matmul(@b)
        (term + @d).inv
      end
    end
  end
end
