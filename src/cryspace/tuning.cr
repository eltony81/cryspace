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

    # Ziegler-Nichols closed-loop tuning rules based on ultimate gain Ku and ultimate period Tu.
    def self.pid_tune_zn(ku : Float64, tu : Float64, type : Symbol = :pid) : NamedTuple(kp: Float64, ki: Float64, kd: Float64)
      raise ArgumentError.new("Ku and Tu must be positive") if ku <= 0.0 || tu <= 0.0
      
      case type
      when :p
        kp = 0.5 * ku
        {kp: kp, ki: 0.0, kd: 0.0}
      when :pi
        kp = 0.45 * ku
        ti = tu / 1.2
        {kp: kp, ki: kp / ti, kd: 0.0}
      else # :pid
        kp = 0.6 * ku
        ti = 0.5 * tu
        td = 0.125 * tu
        {kp: kp, ki: kp / ti, kd: kp * td}
      end
    end

    # Cohen-Coon tuning rules for first-order plus dead-time (FOPDT) models.
    def self.pid_tune_cc(gp : Float64, tau : Float64, theta : Float64, type : Symbol = :pid) : NamedTuple(kp: Float64, ki: Float64, kd: Float64)
      raise ArgumentError.new("Process gain gp, time constant tau, and delay theta must be positive") if gp <= 0.0 || tau <= 0.0 || theta <= 0.0
      
      r = theta / tau
      a = gp * r
      
      case type
      when :p
        kp = (1.0 / a) * (1.0 + r / 3.0)
        {kp: kp, ki: 0.0, kd: 0.0}
      when :pi
        kp = (1.0 / a) * (0.9 + r / 12.0)
        ti = theta * (30.0 + 3.0 * r) / (9.0 + 20.0 * r)
        {kp: kp, ki: kp / ti, kd: 0.0}
      else # :pid
        kp = (1.0 / a) * (1.35 + r / 4.0)
        ti = theta * (32.0 + 6.0 * r) / (13.0 + 8.0 * r)
        td = theta * 4.0 / (11.0 + 2.0 * r)
        {kp: kp, ki: kp / ti, kd: kp * td}
      end
    end
  end
end
