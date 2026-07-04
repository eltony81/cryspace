require "num"

module CrySpace
  module Solver
    extend self

    # Solves an ODE of the form dx/dt = f(x, t) using the Forward Euler method.
    def euler(f : Proc(Float64Tensor, Float64, Float64Tensor), x0 : Float64Tensor, t_span : Tuple(Float64, Float64), dt : Float64)
      t_start, t_end = t_span
      t = t_start
      x = x0.dup
      
      times = [t]
      states = [x]
      
      while t < t_end
        step_dt = dt
        if t + step_dt > t_end
          step_dt = t_end - t
        end
        
        k = f.call(x, t)
        x = x + k * step_dt
        t += step_dt
        
        times << t
        states << x
      end
      
      {times, states}
    end

    # Vectorized Euler: optimized to pre-allocate result matrix and avoid copies
    def euler(f : Proc(Float64Tensor, Float64, Float64Tensor), x0 : Float64Tensor, t_vec : Float64Tensor)
      n_steps = t_vec.size
      n_states = x0.size
      
      x_matrix = Float64Tensor.new([n_steps, n_states])
      copy_to_matrix(x_matrix, 0, x0)
      
      x_current = x0.dup
      
      (n_steps - 1).times do |i|
        t = t_vec[i].value
        dt = t_vec[i + 1].value - t
        
        k = f.call(x_current, t)
        x_current = x_current + k * dt
        copy_to_matrix(x_matrix, i + 1, x_current)
      end
      
      {t_vec, x_matrix}
    end

    # Solves an ODE of the form dx/dt = f(x, t) using the 4th Order Runge-Kutta method.
    def rk4(f : Proc(Float64Tensor, Float64, Float64Tensor), x0 : Float64Tensor, t_span : Tuple(Float64, Float64), dt : Float64)
      t_start, t_end = t_span
      t = t_start
      x = x0.dup
      
      times = [t]
      states = [x]
      
      while t < t_end
        h = dt
        if t + h > t_end
          h = t_end - t
        end
        
        half_h = h / 2.0
        
        k1 = f.call(x, t)
        k2 = f.call(x + k1 * half_h, t + half_h)
        k3 = f.call(x + k2 * half_h, t + half_h)
        k4 = f.call(x + k3 * h, t + h)
        
        x = x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
        t += h
        
        times << t
        states << x
      end
      
      {times, states}
    end

    # Vectorized RK4: optimized to pre-allocate result matrix and avoid copies
    def rk4(f : Proc(Float64Tensor, Float64, Float64Tensor), x0 : Float64Tensor, t_vec : Float64Tensor)
      n_steps = t_vec.size
      n_states = x0.size
      
      x_matrix = Float64Tensor.new([n_steps, n_states])
      copy_to_matrix(x_matrix, 0, x0)
      
      x_current = x0.dup
      
      (n_steps - 1).times do |i|
        t = t_vec[i].value
        h = t_vec[i + 1].value - t
        half_h = h / 2.0
        
        k1 = f.call(x_current, t)
        k2 = f.call(x_current + k1 * half_h, t + half_h)
        k3 = f.call(x_current + k2 * half_h, t + half_h)
        k4 = f.call(x_current + k3 * h, t + h)
        
        x_current = x_current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
        copy_to_matrix(x_matrix, i + 1, x_current)
      end
      
      {t_vec, x_matrix}
    end

    def copy_to_matrix(matrix : Float64Tensor, row : Int, vector : Float64Tensor)
      n = vector.size
      n.times do |j|
        # ALWAYS extract value explicitly from source tensor
        val = vector.rank > 1 ? vector[j, 0].value : vector[j].value
        matrix[row, j] = val
      end
    end

    # Solves an ODE of the form dx/dt = f(x, t) using the Runge-Kutta-Fehlberg 4(5) adaptive step solver.
    def rk45(f : Proc(Float64Tensor, Float64, Float64Tensor), x0 : Float64Tensor, t_span : Tuple(Float64, Float64), tol = 1e-6)
      t_start, t_end = t_span
      t = t_start
      x = x0.dup
      
      times = [t]
      states = [x]
      
      h = (t_end - t_start) / 100.0
      h = 1e-3 if h <= 0.0
      
      h_min = 1e-12
      h_max = (t_end - t_start) / 10.0
      
      while t < t_end
        if t + h > t_end
          h = t_end - t
        end
        
        k1 = f.call(x, t) * h
        k2 = f.call(x + k1 * 0.25, t + h * 0.25) * h
        k3 = f.call(x + k1 * (3.0 / 32.0) + k2 * (9.0 / 32.0), t + h * (3.0 / 8.0)) * h
        k4 = f.call(x + k1 * (1932.0 / 2197.0) - k2 * (7200.0 / 2197.0) + k3 * (7296.0 / 2197.0), t + h * (12.0 / 13.0)) * h
        k5 = f.call(x + k1 * (439.0 / 216.0) - k2 * 8.0 + k3 * (3680.0 / 513.0) - k4 * (845.0 / 4104.0), t + h) * h
        k6 = f.call(x - k1 * (8.0 / 27.0) + k2 * 2.0 - k3 * (3544.0 / 2565.0) + k4 * (1859.0 / 4104.0) - k5 * (11.0 / 40.0), t + h * 0.5) * h
        
        x4 = x + k1 * (25.0 / 216.0) + k3 * (1408.0 / 2565.0) + k4 * (2197.0 / 4104.0) - k5 * 0.2
        x5 = x + k1 * (16.0 / 135.0) + k3 * (6656.0 / 12825.0) + k4 * (28561.0 / 56430.0) - k5 * 0.18 + k6 * (2.0 / 55.0)
        
        err_tensor = x5 - x4
        err = 0.0
        err_tensor.size.times do |i|
          val = err_tensor.rank > 1 ? err_tensor[i, 0].value.abs : err_tensor[i].value.abs
          err = val if val > err
        end
        
        err = 1e-16 if err < 1e-16
        s = 0.84 * ((tol / err) ** 0.25)
        s = [0.1, [s, 4.0].min].max
        
        if err <= tol
          x = x5
          t += h
          times << t
          states << x
          h = [h_min, [h * s, h_max].min].max
        else
          h = [h_min, [h * s, h_max].min].max
          if h <= h_min
            x = x5
            t += h_min
            times << t
            states << x
          end
        end
      end
      
      {times, states}
    end
  end
end
