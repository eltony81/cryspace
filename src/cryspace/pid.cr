# PID Controller with anti-windup and filtered derivative.
module CrySpace
  class PIDController
    property kp : Float64
    property ki : Float64
    property kd : Float64
    property tf : Float64
    property u_min : Float64
    property u_max : Float64

    property integral : Float64 = 0.0
    property prev_error : Float64 = 0.0
    property prev_derivative : Float64 = 0.0

    def initialize(@kp : Float64, @ki : Float64, @kd : Float64, @tf : Float64 = 0.01, @u_min : Float64 = -Float64::INFINITY, @u_max : Float64 = Float64::INFINITY)
    end

    def reset
      @integral = 0.0
      @prev_error = 0.0
      @prev_derivative = 0.0
    end

    # Computes control output u(t) given the error e(t) and time step dt,
    # applying clamping anti-windup to prevent integral windup when saturated.
    def update(error : Float64, dt : Float64) : Float64
      # 1. Proportional term
      p_term = @kp * error
      
      # 2. Derivative term (filtered)
      d_num = @kd * (error - @prev_error) + @tf * @prev_derivative
      d_den = @tf + dt
      d_term = d_num / d_den
      
      # 3. Integral term (tentative update)
      i_term_next = @integral + @ki * error * dt
      
      # Total control signal before saturation
      u_unsat = p_term + i_term_next + d_term
      
      # Apply saturation limits
      u_sat = u_unsat
      if u_unsat > @u_max
        u_sat = @u_max
      elsif u_unsat < @u_min
        u_sat = @u_min
      end
      
      # Clamping Anti-windup logic:
      saturated = u_unsat != u_sat
      same_sign = (error > 0 && u_unsat > 0) || (error < 0 && u_unsat < 0)
      
      if !(saturated && same_sign)
        @integral = i_term_next
      end
      
      # Store variables for next step
      @prev_error = error
      @prev_derivative = d_term
      
      u_sat
    end
  end
end
