require "num"

module CrySpace
  class StateSpace
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
      elsif method == :matched
        to_transferfunction.to_discrete(dt, :matched).to_statespace
      else
        raise ArgumentError.new("Discretization method must be :zoh, :gbt, :bilinear, :tustin, or :matched")
      end
    end

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

    def to_continuous
      dt = @dt
      if dt.nil? || dt <= 0
        raise "System is already continuous"
      end
      
      n = n_states
      m = n_inputs
      
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
  end
end
