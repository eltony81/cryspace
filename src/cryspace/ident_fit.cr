require "num"

module CrySpace
  class Ident
    # Estimates a TransferFunction from input u and output y time series of the given order.
    # Assumes discrete time series with time step dt.
    def self.least_squares_estimation(u : Float64Tensor, y : Float64Tensor, order : Int32, dt : Float64) : TransferFunction
      n = order
      num_samples = u.size
      raise ArgumentError.new("Not enough samples for the requested order") if num_samples <= 2 * n + 1
      
      n_regressors = 2 * n + 1
      n_rows = num_samples - n
      
      phi = Float64Tensor.zeros([n_rows, n_regressors])
      y_vec = Float64Tensor.zeros([n_rows, 1])
      
      n_rows.times do |i|
        k = i + n
        n.times do |j|
          phi[i, j] = -y[k - 1 - j].value
        end
        (n + 1).times do |j|
          phi[i, n + j] = u[k - j].value
        end
        y_vec[i, 0] = y[k].value
      end
      
      phi_t = phi.transpose
      lhs = phi_t.matmul(phi)
      rhs = phi_t.matmul(y_vec)
      
      theta = lhs.solve(rhs)
      
      den_d = Array(Float64).new(n + 1, 0.0)
      den_d[0] = 1.0
      n.times do |j|
        den_d[j + 1] = theta[j, 0].value
      end
      
      num_d = Array(Float64).new(n + 1, 0.0)
      (n + 1).times do |j|
        num_d[j] = theta[n + j, 0].value
      end
      
      TransferFunction.new(num_d.to_tensor, den_d.to_tensor, dt)
    end

    # Realizes a discrete StateSpace model from discrete impulse response data using ERA.
    def self.era(impulse_response : Float64Tensor, n_states : Int32, m_inputs : Int32, n_outputs : Int32, dt : Float64) : StateSpace
      y = if impulse_response.shape.size == 1
            impulse_response.reshape([impulse_response.shape[0], 1, 1])
          elsif impulse_response.shape.size == 2
            impulse_response.reshape([impulse_response.shape[0], impulse_response.shape[1], 1])
          else
            impulse_response
          end
      
      n_samples = y.shape[0]
      p = n_outputs
      m = m_inputs
      nx = n_states
      
      r = (n_samples - 2) // 2
      c = r
      
      h0 = Float64Tensor.zeros([r * p, c * m])
      h1 = Float64Tensor.zeros([r * p, c * m])
      
      r.times do |i|
        c.times do |j|
          y0_block = y[i + j + 1]
          y1_block = y[i + j + 2]
          
          p.times do |pr|
            m.times do |mc|
              h0[i * p + pr, j * m + mc] = y0_block[pr, mc].value
              h1[i * p + pr, j * m + mc] = y1_block[pr, mc].value
            end
          end
        end
      end
      
      u, s, vt = h0.svd
      
      s_n_sqrt = Float64Tensor.zeros([nx, nx])
      s_n_inv_sqrt = Float64Tensor.zeros([nx, nx])
      nx.times do |i|
        val = s[i].value
        val = 1e-12 if val <= 0.0
        s_n_sqrt[i, i] = Math.sqrt(val)
        s_n_inv_sqrt[i, i] = 1.0 / Math.sqrt(val)
      end
      
      u_n = Float64Tensor.zeros([r * p, nx])
      v_n = Float64Tensor.zeros([c * m, nx])
      
      (r * p).times do |i|
        nx.times do |j|
          u_n[i, j] = u[i, j].value
        end
      end
      
      (c * m).times do |i|
        nx.times do |j|
          v_n[i, j] = vt[j, i].value
        end
      end
      
      ad = s_n_inv_sqrt.matmul(u_n.transpose).matmul(h1).matmul(v_n).matmul(s_n_inv_sqrt)
      
      v_n_t_m = Float64Tensor.zeros([nx, m])
      nx.times do |i|
        m.times do |j|
          v_n_t_m[i, j] = v_n[j, i].value
        end
      end
      bd = s_n_sqrt.matmul(v_n_t_m)
      
      u_n_p = Float64Tensor.zeros([p, nx])
      p.times do |i|
        nx.times do |j|
          u_n_p[i, j] = u_n[i, j].value
        end
      end
      cd = u_n_p.matmul(s_n_sqrt)
      
      dd = Float64Tensor.zeros([p, m])
      p.times do |pr|
        m.times do |mc|
          dd[pr, mc] = y[0][pr, mc].value
        end
      end
      
      StateSpace.new(ad, bd, cd, dd, dt)
    end

    # Realizes a discrete StateSpace model from discrete impulse response data using Ho-Kalman.
    def self.ho_kalman(impulse_response : Float64Tensor, n_states : Int32, m_inputs : Int32, n_outputs : Int32, dt : Float64) : StateSpace
      era(impulse_response, n_states, m_inputs, n_outputs, dt)
    end

    # Realizes a discrete StateSpace model using Kung's realization algorithm.
    def self.kung(impulse_response : Float64Tensor, n_states : Int32, m_inputs : Int32, n_outputs : Int32, dt : Float64) : StateSpace
      y = if impulse_response.shape.size == 1
            impulse_response.reshape([impulse_response.shape[0], 1, 1])
          elsif impulse_response.shape.size == 2
            impulse_response.reshape([impulse_response.shape[0], impulse_response.shape[1], 1])
          else
            impulse_response
          end
      
      n_samples = y.shape[0]
      p = n_outputs
      m = m_inputs
      nx = n_states
      
      r = (n_samples - 2) // 2
      c = r
      
      h0 = Float64Tensor.zeros([r * p, c * m])
      r.times do |i|
        c.times do |j|
          y0_block = y[i + j + 1]
          p.times do |pr|
            m.times do |mc|
              h0[i * p + pr, j * m + mc] = y0_block[pr, mc].value
            end
          end
        end
      end
      
      u, s, vt = h0.svd
      
      s_n_sqrt = Float64Tensor.zeros([nx, nx])
      nx.times do |i|
        val = s[i].value
        s_n_sqrt[i, i] = Math.sqrt([val, 1e-12].max)
      end
      
      u_n = Float64Tensor.zeros([r * p, nx])
      (r * p).times do |i|
        nx.times do |j|
          u_n[i, j] = u[i, j].value
        end
      end
      
      obs = u_n.matmul(s_n_sqrt)
      
      obs1 = Float64Tensor.zeros([(r - 1) * p, nx])
      obs2 = Float64Tensor.zeros([(r - 1) * p, nx])
      ((r - 1) * p).times do |i|
        nx.times do |j|
          obs1[i, j] = obs[i, j].value
          obs2[i, j] = obs[p + i, j].value
        end
      end
      
      obs1_t = obs1.transpose
      ad = (obs1_t.matmul(obs1)).inv.matmul(obs1_t).matmul(obs2)
      
      cd = Float64Tensor.zeros([p, nx])
      p.times do |i|
        nx.times do |j|
          cd[i, j] = obs[i, j].value
        end
      end
      
      v_n_t = Float64Tensor.zeros([nx, c * m])
      nx.times do |i|
        (c * m).times do |j|
          v_n_t[i, j] = vt[i, j].value
        end
      end
      bd_all = s_n_sqrt.matmul(v_n_t)
      bd = Float64Tensor.zeros([nx, m])
      nx.times do |i|
        m.times do |j|
          bd[i, j] = bd_all[i, j].value
        end
      end
      
      dd = Float64Tensor.zeros([p, m])
      p.times do |pr|
        m.times do |mc|
          dd[pr, mc] = y[0][pr, mc].value
        end
      end
      
      StateSpace.new(ad, bd, cd, dd, dt)
    end

    # Fits a transfer function coefficients directly to frequency response data.
    def self.levy_fit(omega : Float64Tensor, resp : Tensor(Complex, CPU(Complex)), m_num : Int32, n_den : Int32) : TransferFunction
      n_freqs = omega.size
      n_vars = m_num + 1 + n_den
      
      phi = Float64Tensor.zeros([2 * n_freqs, n_vars])
      y_vec = Float64Tensor.zeros([2 * n_freqs, 1])
      
      n_freqs.times do |i|
        w = omega[i].value
        r = resp[i].value.real
        img = resp[i].value.imag
        
        (m_num + 1).times do |k|
          if k % 4 == 0
            phi[2 * i, k] = w ** k
          elsif k % 4 == 2
            phi[2 * i, k] = -(w ** k)
          else
            phi[2 * i, k] = 0.0
          end
        end
        n_den.times do |k|
          pow = k + 1
          if pow % 4 == 1
            phi[2 * i, m_num + 1 + k] = (w ** pow) * img
          elsif pow % 4 == 2
            phi[2 * i, m_num + 1 + k] = (w ** pow) * r
          elsif pow % 4 == 3
            phi[2 * i, m_num + 1 + k] = -(w ** pow) * img
          else
            phi[2 * i, m_num + 1 + k] = -(w ** pow) * r
          end
        end
        y_vec[2 * i, 0] = r
        
        (m_num + 1).times do |k|
          if k % 4 == 1
            phi[2 * i + 1, k] = w ** k
          elsif k % 4 == 3
            phi[2 * i + 1, k] = -(w ** k)
          else
            phi[2 * i + 1, k] = 0.0
          end
        end
        n_den.times do |k|
          pow = k + 1
          if pow % 4 == 1
            phi[2 * i + 1, m_num + 1 + k] = -(w ** pow) * r
          elsif pow % 4 == 2
            phi[2 * i + 1, m_num + 1 + k] = (w ** pow) * img
          elsif pow % 4 == 3
            phi[2 * i + 1, m_num + 1 + k] = (w ** pow) * r
          else
            phi[2 * i + 1, m_num + 1 + k] = -(w ** pow) * img
          end
        end
        y_vec[2 * i + 1, 0] = img
      end
      
      phi_t = phi.transpose
      lhs = phi_t.matmul(phi)
      rhs = phi_t.matmul(y_vec)
      
      theta = lhs.solve(rhs)
      
      num_arr = Array(Float64).new(m_num + 1, 0.0)
      (m_num + 1).times do |k|
        num_arr[m_num - k] = theta[k, 0].value
      end
      
      den_arr = Array(Float64).new(n_den + 1, 0.0)
      den_arr[n_den] = 1.0
      n_den.times do |k|
        den_arr[n_den - 1 - k] = theta[m_num + 1 + k, 0].value
      end
      
      TransferFunction.new(num_arr.to_tensor, den_arr.to_tensor)
    end
  end
end
