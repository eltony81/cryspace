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

    # Solves the Continuous Algebraic Riccati Equation:
    # A^T * P + P * A - P * B * R^-1 * B^T * P + Q = 0
    # Returns P.
    def care(q : Float64Tensor, r : Float64Tensor)
      n = n_states
      g = @b.matmul(r.inv).matmul(@b.transpose)

      # Construct Hamiltonian matrix H = [A -G; -Q -A^T]
      # Using Num.vstack/hstack for performance
      h_top = Num.hstack(@a, -g)
      h_bottom = Num.hstack(-q, -@a.transpose)
      h = Num.vstack(h_top, h_bottom)

      # Use Schur decomposition for better stability if available, 
      # otherwise fallback to eigenvalue decomposition (Potter's method)
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
      Tensor.lyapunov(@a, -q)
    end

    # Solves the Discrete-Time Algebraic Riccati Equation (DARE) using Schur method:
    # A^T * P * A - P - A^T * P * B * (R + B^T * P * B)^-1 * B^T * P * A + Q = 0
    # Returns P.
    def dare(q : Float64Tensor, r : Float64Tensor)
      n = n_states
      m = n_inputs
      
      # G = B * R^-1 * B^T
      g = @b.matmul(r.inv).matmul(@b.transpose)
      
      # We use the symplectic pencil (L, M) for the generalized eigenvalue problem
      # L = [A 0; -Q I], M = [I G; 0 A^T]
      # Using Num.vstack/hstack for performance
      eye = Float64Tensor.identity(n)
      zeros = Float64Tensor.zeros([n, n])
      
      l_top = Num.hstack(@a, zeros)
      l_bottom = Num.hstack(-q, eye)
      l = Num.vstack(l_top, l_bottom)
      
      m_top = Num.hstack(eye, g)
      m_bottom = Num.hstack(zeros, @a.transpose)
      m_mat = Num.vstack(m_top, m_bottom)
      
      # For now, we use eig_c on M^-1 * L if M is invertible
      # In a future version of num.cr we could use gges
      begin
        s_mat = m_mat.inv.matmul(l)
        w, v = s_mat.eig_c
      rescue
        # Fallback to iterative if inversion fails
        return dare_iterative(q, r)
      end
      
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
        if w.to_unsafe[i].abs < 1.0
          stable_indices << i
        end
      end

      if stable_indices.size != n
        # Try iterative as fallback
        return dare_iterative(q, r)
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

    private def dare_iterative(q : Float64Tensor, r : Float64Tensor, max_iter = 1000, tol = 1e-9)
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
      # Check if num.cr has discrete_lyapunov, otherwise use kron fallback
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

    # Computes normalized coprime factorization G = N * M^-1
    def coprime_factorization : Tuple(StateSpace, StateSpace)
      q = Float64Tensor.identity(n_states)
      r = Float64Tensor.identity(n_inputs)
      k_gain, _, _ = lqr(q, r)
      a_cl = @a - @b.matmul(k_gain)
      m_sys = StateSpace.new(a_cl, @b, -k_gain, Float64Tensor.identity(n_inputs), @dt)
      n_sys = StateSpace.new(a_cl, @b, @c - @d.matmul(k_gain), @d, @dt)
      {n_sys, m_sys}
    end

    # Computes static decouple gain matrix W (where G(0)*W is decoupled/diagonal)
    def decouple_gain : Float64Tensor
      dc = dcgain
      dc.shape[0] == dc.shape[1] ? dc.inv : dc.transpose
    end
  end
end
