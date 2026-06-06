require "num"

module CrySpace
  class AdaptiveNonlinear
    # Computes describing function value for a backlash nonlinearity of half-width b at input amplitude a.
    def self.describing_function_backlash(a : Float64, b : Float64) : Complex
      raise ArgumentError.new("Amplitude a must be positive") if a <= 0.0
      raise ArgumentError.new("Backlash half-width b must be positive") if b <= 0.0
      
      if a <= b
        return Complex.new(0.0, 0.0)
      end
      
      ratio = 1.0 - 2.0 * b / a
      x = Math.asin(ratio)
      n_real = (1.0 / Math::PI) * (Math::PI / 2.0 + x + ratio * Math.cos(x))
      n_imag = -(4.0 * b / (Math::PI * a)) * (1.0 - b / a)
      
      Complex.new(n_real, n_imag)
    end

    # Performs trajectory optimization for nonlinear systems using Iterative LQR (iLQR).
    def self.ilqr(
      f : Proc(Float64Tensor, Float64Tensor, Float64Tensor),
      jf : Proc(Float64Tensor, Float64Tensor, Float64Tensor),
      ju : Proc(Float64Tensor, Float64Tensor, Float64Tensor),
      q : Float64Tensor,
      r : Float64Tensor,
      qf : Float64Tensor,
      x0 : Float64Tensor,
      u_init : Array(Float64Tensor),
      max_iter : Int32 = 5
    ) : Tuple(Array(Float64Tensor), Array(Float64Tensor))
      n_steps = u_init.size
      nx = x0.shape[0]
      nu = u_init[0].shape[0]
      
      u = u_init.map(&.dup)
      x = Array(Float64Tensor).new(n_steps + 1)
      x << x0.dup
      
      n_steps.times do |k|
        x << f.call(x[k], u[k])
      end
      
      max_iter.times do
        v = qf.matmul(x.last)
        v_cov = qf.dup
        
        k_feedbacks = Array(Float64Tensor).new(n_steps)
        d_feedforwards = Array(Float64Tensor).new(n_steps)
        
        (n_steps - 1).downto(0) do |k|
          xk = x[k]
          uk = u[k]
          
          a = jf.call(xk, uk)
          b = ju.call(xk, uk)
          
          gx = q.matmul(xk)
          cx = q.dup
          gu = r.matmul(uk)
          cu = r.dup
          
          q_x = gx + a.transpose.matmul(v)
          q_u = gu + b.transpose.matmul(v)
          q_xx = cx + a.transpose.matmul(v_cov).matmul(a)
          q_uu = cu + b.transpose.matmul(v_cov).matmul(b)
          q_ux = b.transpose.matmul(v_cov).matmul(a)
          
          q_uu_inv = q_uu.inv
          k_gain = -q_uu_inv.matmul(q_ux)
          d_gain = -q_uu_inv.matmul(q_u)
          
          k_feedbacks << k_gain
          d_feedforwards << d_gain
          
          v = q_x + k_gain.transpose.matmul(q_uu).matmul(d_gain) + k_gain.transpose.matmul(q_u) + q_ux.transpose.matmul(d_gain)
          v_cov = q_xx + k_gain.transpose.matmul(q_uu).matmul(k_gain) + k_gain.transpose.matmul(q_ux) + q_ux.transpose.matmul(k_gain)
        end
        k_feedbacks.reverse!
        d_feedforwards.reverse!
        
        x_new = Array(Float64Tensor).new(n_steps + 1)
        x_new << x0.dup
        u_new = Array(Float64Tensor).new(n_steps)
        
        n_steps.times do |k|
          dx = x_new[k] - x[k]
          un = u[k] + d_feedforwards[k] + k_feedbacks[k].matmul(dx)
          u_new << un
          x_new << f.call(x_new[k], un)
        end
        
        x = x_new
        u = u_new
      end
      
      {x, u}
    end

    # Simulates Model Reference Adaptive Control (MRAC) tracking a reference model.
    def self.mrac_simulate(
      a_plant : Float64,
      b_plant : Float64,
      a_model : Float64,
      b_model : Float64,
      gamma_x : Float64,
      gamma_r : Float64,
      r_signal : Proc(Float64, Float64),
      t_vec : Float64Tensor
    ) : Tuple(Float64Tensor, Float64Tensor, Float64Tensor)
      n_steps = t_vec.size
      dt = t_vec[1].value - t_vec[0].value
      
      x = Float64Tensor.zeros([n_steps])
      xm = Float64Tensor.zeros([n_steps])
      u = Float64Tensor.zeros([n_steps])
      
      x_val = 0.0
      xm_val = 0.0
      theta_x = 0.0
      theta_r = 0.0
      
      sgn = b_plant > 0.0 ? 1 : (b_plant < 0.0 ? -1 : 0)
      
      n_steps.times do |i|
        t = t_vec[i].value
        r = r_signal.call(t)
        
        x[i] = x_val
        xm[i] = xm_val
        
        u_val = theta_x * x_val + theta_r * r
        u[i] = u_val
        
        e = x_val - xm_val
        
        dx = a_plant * x_val + b_plant * u_val
        dxm = a_model * xm_val + b_model * r
        
        d_theta_x = -gamma_x * e * x_val * sgn
        d_theta_r = -gamma_r * e * r * sgn
        
        x_val += dx * dt
        xm_val += dxm * dt
        theta_x += d_theta_x * dt
        theta_r += d_theta_r * dt
      end
      {x, xm, u}
    end
  end
end
