module CrySpace
  class Nonlinear
    # Computes describing function value for a saturation nonlinearity of limit s at input amplitude a.
    def self.describing_function_saturation(a : Float64, s : Float64) : Float64
      raise ArgumentError.new("Amplitude a must be positive") if a <= 0.0
      raise ArgumentError.new("Saturation limit s must be positive") if s <= 0.0
      
      if a <= s
        return 1.0
      end
      
      ratio = s / a
      (2.0 / Math::PI) * (Math.asin(ratio) + ratio * Math.sqrt(1.0 - ratio * ratio))
    end

    # Computes describing function value for a deadzone nonlinearity of half-width d at input amplitude a.
    def self.describing_function_deadzone(a : Float64, d : Float64) : Float64
      raise ArgumentError.new("Amplitude a must be positive") if a <= 0.0
      raise ArgumentError.new("Deadzone half-width d must be positive") if d <= 0.0
      
      if a <= d
        return 0.0
      end
      
      ratio = d / a
      1.0 - (2.0 / Math::PI) * (Math.asin(ratio) + ratio * Math.sqrt(1.0 - ratio * ratio))
    end
  end
end
