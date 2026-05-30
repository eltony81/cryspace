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

    private def copy_to_matrix(matrix : Float64Tensor, row : Int, vector : Float64Tensor)
      n = vector.size
      m_ptr = matrix.to_unsafe + (row * n)
      v_ptr = vector.to_unsafe
      n.times do |j|
        m_ptr[j] = v_ptr[j]
      end
    end
  end
end
