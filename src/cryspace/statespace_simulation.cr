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
  end
end
