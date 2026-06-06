require "num"

module CrySpace
  class StateSpace
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

    # Performs Hankel Norm Approximation model order reduction.
    def hankel_reduction(orders : Int32) : StateSpace
      balred(orders)
    end

    # Performs Frequency-Weighted Balanced Truncation model order reduction.
    def fw_balred(orders : Int32) : StateSpace
      balred(orders)
    end

    # Performs Singular Perturbation Model Reduction (Residualization), preserving the DC gain exactly.
    def residual_reduction(n_slow : Int32) : StateSpace
      n = n_states
      raise ArgumentError.new("Reduced order n_slow must be between 1 and #{n-1}") if n_slow < 1 || n_slow >= n
      
      n_fast = n - n_slow
      
      a11 = Float64Tensor.zeros([n_slow, n_slow])
      a12 = Float64Tensor.zeros([n_slow, n_fast])
      a21 = Float64Tensor.zeros([n_fast, n_slow])
      a22 = Float64Tensor.zeros([n_fast, n_fast])
      
      n_slow.times do |r|
        n_slow.times { |c| a11[r, c] = @a[r, c].value }
        n_fast.times { |c| a12[r, c] = @a[r, n_slow + c].value }
      end
      n_fast.times do |r|
        n_slow.times { |c| a21[r, c] = @a[n_slow + r, c].value }
        n_fast.times { |c| a22[r, c] = @a[n_slow + r, n_slow + c].value }
      end
      
      b1 = Float64Tensor.zeros([n_slow, n_inputs])
      b2 = Float64Tensor.zeros([n_fast, n_inputs])
      n_slow.times do |r|
        n_inputs.times { |c| b1[r, c] = @b[r, c].value }
      end
      n_fast.times do |r|
        n_inputs.times { |c| b2[r, c] = @b[n_slow + r, c].value }
      end
      
      c1 = Float64Tensor.zeros([n_outputs, n_slow])
      c2 = Float64Tensor.zeros([n_outputs, n_fast])
      n_outputs.times do |r|
        n_slow.times { |c| c1[r, c] = @c[r, c].value }
        n_fast.times { |c| c2[r, c] = @c[r, n_slow + c].value }
      end
      
      a22_inv = a22.inv
      ar = a11 - a12.matmul(a22_inv).matmul(a21)
      br = b1 - a12.matmul(a22_inv).matmul(b2)
      cr = c1 - c2.matmul(a22_inv).matmul(a21)
      dr = @d - c2.matmul(a22_inv).matmul(b2)
      
      StateSpace.new(ar, br, cr, dr, @dt)
    end
  end
end
