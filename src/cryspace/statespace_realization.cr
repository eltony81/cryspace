require "num"

module CrySpace
  class StateSpace
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

      m_mat = Float64Tensor.zeros([n, n])
      n.times do |i|
        n.times do |j|
          if j >= i
            m_mat[i, j] = poly[j - i]
          end
        end
      end

      t_inv = m_mat.matmul(ob)
      t_matrix = t_inv.inv
      {similarity_transform(t_matrix), t_matrix}
    end

    def to_transferfunction
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

    def to_modal_form
      n = n_states
      w, v = @a.eig_c
      
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

    def balred(orders : Int32)
      if orders > n_states
        raise ArgumentError.new("Reduced order must be less than or equal to current order")
      end
      if orders <= 0
        raise ArgumentError.new("Reduced order must be greater than 0")
      end
      
      wc = gram(:c)
      wo = gram(:o)
      
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
      
      product = lo.transpose.matmul(lc)
      u_p, s_p, vt_p = product.svd
      v_p = vt_p.transpose
      
      t_matrix = Float64Tensor.zeros([n, n])
      t_inv = Float64Tensor.zeros([n, n])
      
      n.times do |i|
        sig_val = s_p[i].value
        sig_factor = sig_val.abs > 1e-12 ? 1.0 / Math.sqrt(sig_val) : 0.0
        
        n.times do |r|
          sum = 0.0
          n.times do |k|
            sum += lc[r, k].value * v_p[k, i].value
          end
          t_matrix[r, i] = sum * sig_factor
        end
        
        n.times do |c_idx|
          sum = 0.0
          n.times do |k|
            sum += u_p[k, i].value * lo[c_idx, k].value
          end
          t_inv[i, c_idx] = sum * sig_factor
        end
      end
      
      ab = t_inv.matmul(@a).matmul(t_matrix)
      bb = t_inv.matmul(@b)
      cb = @c.matmul(t_matrix)
      
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

    def similarity_transform(t_matrix : Float64Tensor) : StateSpace
      t_inv = t_matrix.inv
      a_new = t_inv.matmul(@a).matmul(t_matrix)
      b_new = t_inv.matmul(@b)
      c_new = @c.matmul(t_matrix)
      StateSpace.new(a_new, b_new, c_new, @d, @dt)
    end

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

    def minreal : StateSpace
      n = n_states
      return self if n == 0
      
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

    def ss2tf : TransferFunction
      to_transferfunction
    end

    def gram(type : Symbol)
      if type == :c
        q = @b.matmul(@b.transpose)
        lyap(q)
      elsif type == :o
        q = @c.transpose.matmul(@c)
        sys_t = StateSpace.new(@a.transpose, @b, @c, @d)
        sys_t.lyap(q)
      else
        raise ArgumentError.new("Gramian type must be :c (controllability) or :o (observability)")
      end
    end

    def hsvd
      wc = gram(:c)
      wo = gram(:o)
      prod = wc.matmul(wo)
      eigvals = prod.eigvals
      res = Array(Float64).new(eigvals.size)
      eigvals.size.times do |i|
        res << Math.sqrt(eigvals.to_unsafe[i].abs)
      end
      res.sort.reverse
    end

    def ctrb : Float64Tensor
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

    def obsv : Float64Tensor
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

    # Computes the transmission zeros of the system.
    def transmission_zeros : Array(Complex)
      if n_inputs == 1 && n_outputs == 1
        to_transferfunction.zeros
      else
        raise NotImplementedError.new("MIMO transmission zeros calculation not supported yet")
      end
    end

    # Computes a balanced realization of the system.
    # Returns: {balanced_system, t_matrix, t_inv}
    def balreal : Tuple(StateSpace, Float64Tensor, Float64Tensor)
      wc = gram(:c)
      wo = gram(:o)
      
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
      
      product = lo.transpose.matmul(lc)
      u_p, s_p, vt_p = product.svd
      v_p = vt_p.transpose
      
      t_matrix = Float64Tensor.zeros([n, n])
      t_inv = Float64Tensor.zeros([n, n])
      
      n.times do |i|
        sig_val = s_p[i].value
        sig_factor = sig_val.abs > 1e-12 ? 1.0 / Math.sqrt(sig_val) : 0.0
        
        n.times do |r|
          sum = 0.0
          n.times do |k|
            sum += lc[r, k].value * v_p[k, i].value
          end
          t_matrix[r, i] = sum * sig_factor
        end
        
        n.times do |c_idx|
          sum = 0.0
          n.times do |k|
            sum += u_p[k, i].value * lo[c_idx, k].value
          end
          t_inv[i, c_idx] = sum * sig_factor
        end
      end
      
      ab = t_inv.matmul(@a).matmul(t_matrix)
      bb = t_inv.matmul(@b)
      cb = @c.matmul(t_matrix)
      
      {StateSpace.new(ab, bb, cb, @d, @dt), t_matrix, t_inv}
    end
  end
end
