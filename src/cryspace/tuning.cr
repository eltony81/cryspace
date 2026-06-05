# Ziegler-Nichols tuning rules for PID controllers in CrySpace.
module CrySpace
  module Tuning
    # Ziegler-Nichols open-loop tuning rules for a First-Order Plus Dead-Time (FOPDT) model:
    # G(s) = (K * e^(-theta * s)) / (tau * s + 1)
    # Returns: {kp, ki, kd}
    def self.ziegler_nichols_fopdt(k : Float64, tau : Float64, theta : Float64)
      raise ArgumentError.new("Dead time theta must be positive") if theta <= 0.0
      
      kp = (1.2 * tau) / (k * theta)
      ti = 2.0 * theta
      td = 0.5 * theta
      
      ki = kp / ti
      kd = kp * td
      
      {kp, ki, kd}
    end
  end
end
