require "num"

module CrySpace
  module Solver
    extend self

    # Solves an ODE of the form dx/dt = f(x, t) using the Forward Euler method.
    # f: A Proc that takes the current state (Tensor) and time (Float64) and returns the derivative (Tensor).
    # x0: Initial state vector.
    # t_span: A tuple (t_start, t_end).
    # dt: Time step.
    def euler(f : Proc(Float64Tensor, Float64, Float64Tensor), x0 : Float64Tensor, t_span : Tuple(Float64, Float64), dt : Float64)
      t_start, t_end = t_span
      t = t_start
      x = x0.dup
      
      times = [t]
      states = [x]
      
      while t < t_end
        # Ensure we don't overshoot t_end
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

    # Solves an ODE of the form dx/dt = f(x, t) using the 4th Order Runge-Kutta method.
    # f: A Proc that takes the current state (Tensor) and time (Float64) and returns the derivative (Tensor).
    # x0: Initial state vector.
    # t_span: A tuple (t_start, t_end).
    # dt: Time step.
    def rk4(f : Proc(Float64Tensor, Float64, Float64Tensor), x0 : Float64Tensor, t_span : Tuple(Float64, Float64), dt : Float64)
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
        
        h = step_dt
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
  end
end
