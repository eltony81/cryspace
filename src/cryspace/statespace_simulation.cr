require "num"

module CrySpace
  class StateSpace
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

    # Simulates impulse response of discrete or continuous (sampled) system.
    def impulse_response(n_steps = 100)
      sys = self
      curr_dt = @dt
      unless curr_dt && curr_dt > 0
        sys = self.sample(0.1)
      end
      
      dt = sys.dt.not_nil!
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

    # Simulates unforced free response with non-zero initial state x0.
    def initial_response(x0 : Float64Tensor, n_steps = 100)
      sys = self
      curr_dt = @dt
      unless curr_dt && curr_dt > 0
        sys = self.sample(0.1)
      end
      
      dt = sys.dt.not_nil!
      x = x0.dup
      
      t_arr = Array(Float64).new(n_steps)
      x_arr = Array(Float64Tensor).new(n_steps)
      y_arr = Array(Float64Tensor).new(n_steps)
      
      n_steps.times do |i|
        t_arr << i * dt
        x_arr << x
        y = sys.c.matmul(x)
        y_arr << y
        x = sys.a.matmul(x)
      end
      
      {t_arr, x_arr, y_arr}
    end

    # Simulates time response with arbitrary input (wrapper for simulate).
    def lsim(u : Float64Tensor, t : Float64Tensor, x0 : Float64Tensor? = nil, method = :rk4)
      simulate(t, x0: x0, u: u, method: method)
    end

    struct StepInfo
      property rise_time : Float64
      property settling_time : Float64
      property overshoot : Float64
      property peak : Float64
      property peak_time : Float64
      property steady_state_value : Float64

      def initialize(@rise_time, @settling_time, @overshoot, @peak, @peak_time, @steady_state_value)
      end
    end

    # Analyzes the step response of a SISO system and returns step response performance metrics.
    def stepinfo(n_steps = 500, settling_threshold = 0.02) : StepInfo
      unless n_inputs == 1 && n_outputs == 1
        raise ArgumentError.new("stepinfo only supported for SISO systems")
      end

      # First run a step response simulation
      t, _, y = step_response(n_steps)
      
      # Convert y to Array(Float64)
      y_siso = y.map { |yi| yi[0, 0].value }
      
      # Final value is our steady-state estimation.
      # Average the last 10% to smooth out oscillations if any.
      last_n = (n_steps * 0.1).to_i
      last_n = 1 if last_n < 1
      y_ss = 0.0
      last_n.times { |i| y_ss += y_siso[n_steps - 1 - i] }
      y_ss /= last_n
      
      y_max = y_siso.max
      y_min = y_siso.min
      
      # Find peak
      peak = y_max
      if y_min.abs > y_max.abs
        peak = y_min
      end
      
      peak_idx = y_siso.index(peak) || 0
      peak_time = t[peak_idx]
      
      overshoot = 0.0
      if y_ss.abs > 1e-9
        overshoot = 100.0 * (peak - y_ss) / y_ss
        overshoot = 0.0 if overshoot < 0.0
      end
      
      # Rise time: 10% to 90% of steady-state value
      t_10 = nil
      t_90 = nil
      
      n_steps.times do |i|
        val = y_siso[i]
        if y_ss >= 0
          if t_10.nil? && val >= 0.1 * y_ss
            t_10 = t[i]
          end
          if t_90.nil? && val >= 0.9 * y_ss
            t_90 = t[i]
          end
        else
          if t_10.nil? && val <= 0.1 * y_ss
            t_10 = t[i]
          end
          if t_90.nil? && val <= 0.9 * y_ss
            t_90 = t[i]
          end
        end
      end
      
      rise_time = if t_10 && t_90
        t_90 - t_10
      else
        0.0
      end
      
      # Settling time: search backwards for the last time the output is outside the threshold band
      err_bound = (y_ss * settling_threshold).abs
      settling_idx = 0
      (n_steps - 1).downto(0) do |i|
        if (y_siso[i] - y_ss).abs > err_bound
          settling_idx = i + 1
          break
        end
      end
      
      settling_idx = n_steps - 1 if settling_idx >= n_steps
      settling_time = t[settling_idx]
      
      StepInfo.new(rise_time, settling_time, overshoot, peak, peak_time, y_ss)
    end

    # Simulates coupled state-feedback controller with an observer/estimator.
    # Optionally adds process and measurement Gaussian noises under covariance matrices.
    def simulate_observer(
      t : Float64Tensor,
      k_gain : Float64Tensor,
      l_gain : Float64Tensor,
      x0 : Float64Tensor? = nil,
      x0_est : Float64Tensor? = nil,
      u_ref : Float64Tensor? = nil,
      process_noise_cov : Float64Tensor? = nil,
      measure_noise_cov : Float64Tensor? = nil
    )
      n_steps = t.size
      n = n_states
      m = n_inputs
      p = n_outputs
      
      x_current = x0.nil? ? Float64Tensor.zeros([n, 1]) : x0.dup
      x_est_current = x0_est.nil? ? Float64Tensor.zeros([n, 1]) : x0_est.dup
      u_ref_val = u_ref.nil? ? Float64Tensor.zeros([m, 1]) : u_ref
      
      x_matrix = Float64Tensor.new([n_steps, n])
      x_est_matrix = Float64Tensor.new([n_steps, n])
      y_matrix = Float64Tensor.new([n_steps, p])
      u_matrix = Float64Tensor.new([n_steps, m])
      
      randn = ->() {
        u1 = Random.rand
        u2 = Random.rand
        Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
      }
      
      cholesky = ->(cov : Float64Tensor) {
        dim = cov.shape[0]
        l = Float64Tensor.zeros([dim, dim])
        is_diagonal = true
        dim.times do |r|
          dim.times do |c|
            if r != c && cov[r, c].value.abs > 1e-9
              is_diagonal = false
            end
          end
        end
        
        if is_diagonal
          dim.times do |r|
            l[r, r] = Math.sqrt({cov[r, r].value, 0.0}.max)
          end
        else
          dim.times do |i|
            (i + 1).times do |j|
              s = 0.0
              j.times do |k|
                s += l[i, k].value * l[j, k].value
              end
              if i == j
                val = cov[i, i].value - s
                l[i, j] = Math.sqrt(val > 0.0 ? val : 0.0)
              else
                l_jj = l[j, j].value
                l[i, j] = l_jj > 1e-12 ? (cov[i, j].value - s) / l_jj : 0.0
              end
            end
          end
        end
        l
      }
      
      w_l = process_noise_cov.nil? ? nil : cholesky.call(process_noise_cov)
      v_l = measure_noise_cov.nil? ? nil : cholesky.call(measure_noise_cov)
      
      n_steps.times do |i|
        u_ref_step = Float64Tensor.zeros([m, 1])
        if u_ref_val.rank == 2 && u_ref_val.shape[1] == n_steps
          m.times { |r| u_ref_step[r, 0] = u_ref_val[r, i] }
        else
          u_ref_step = u_ref_val
        end
        
        u = u_ref_step - k_gain.matmul(x_est_current)
        
        w = Float64Tensor.zeros([n, 1])
        if w_l
          n.times do |r|
            w[r, 0] = randn.call
          end
          w = w_l.matmul(w)
        end
        
        v = Float64Tensor.zeros([p, 1])
        if v_l
          p.times do |r|
            v[r, 0] = randn.call
          end
          v = v_l.matmul(v)
        end
        
        y = @c.matmul(x_current) + @d.matmul(u) + v
        
        n.times { |j| x_matrix[i, j] = x_current[j, 0].value }
        n.times { |j| x_est_matrix[i, j] = x_est_current[j, 0].value }
        p.times { |j| y_matrix[i, j] = y[j, 0].value }
        m.times { |j| u_matrix[i, j] = u[j, 0].value }
        
        if i < n_steps - 1
          h = t.to_unsafe[i + 1] - t.to_unsafe[i]
          
          dx = @a.matmul(x_current) + @b.matmul(u)
          dy_err = y - @c.matmul(x_est_current) - @d.matmul(u)
          dx_est = @a.matmul(x_est_current) + @b.matmul(u) + l_gain.matmul(dy_err)
          
          if w_l
            x_current = x_current + dx * h + w * Math.sqrt(h)
          else
            x_current = x_current + dx * h
          end
          
          x_est_current = x_est_current + dx_est * h
        end
      end
      
      {t, x_matrix, x_est_matrix, y_matrix, u_matrix}
    end

    # Simulates ramp response of discrete or continuous (sampled) system.
    def ramp_response(n_steps = 100)
      sys = self
      curr_dt = @dt
      unless curr_dt && curr_dt > 0
        sys = self.sample(0.1)
      end
      
      dt = sys.dt.not_nil!
      x = Float64Tensor.zeros([sys.n_states, 1])
      
      t_arr = Array(Float64).new(n_steps)
      x_arr = Array(Float64Tensor).new(n_steps)
      y_arr = Array(Float64Tensor).new(n_steps)
      
      n_steps.times do |i|
        t_val = i * dt
        t_arr << t_val
        u = Float64Tensor.zeros([sys.n_inputs, 1])
        sys.n_inputs.times { |idx| u[idx, 0] = t_val }
        
        x_arr << x
        y = sys.c.matmul(x) + sys.d.matmul(u)
        y_arr << y
        x = sys.a.matmul(x) + sys.b.matmul(u)
      end
      
      {t_arr, x_arr, y_arr}
    end
  end
end
