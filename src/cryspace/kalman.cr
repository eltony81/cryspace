# Discrete Kalman Filter implementation for StateSpace systems in CrySpace.
require "num"

module CrySpace
  class KalmanFilter
    property sys : StateSpace
    property q : Float64Tensor
    property r : Float64Tensor
    property x : Float64Tensor
    property p : Float64Tensor

    def initialize(@sys : StateSpace, @q : Float64Tensor, @r : Float64Tensor, x0 : Float64Tensor? = nil, p0 : Float64Tensor? = nil)
      unless @sys.dt && @sys.dt.not_nil! > 0
        raise ArgumentError.new("Kalman filter requires a discrete-time system. Discretize your system using sys.sample(dt) first.")
      end
      
      n = @sys.n_states
      @x = x0 || Float64Tensor.zeros([n, 1])
      @p = p0 || Float64Tensor.identity(n)
    end

    def predict(u : Float64Tensor? = nil)
      u_val = u || Float64Tensor.zeros([@sys.n_inputs, 1])
      
      # State prediction: x = A * x + B * u
      @x = @sys.a.matmul(@x) + @sys.b.matmul(u_val)
      
      # Covariance prediction: P = A * P * A^T + Q
      @p = @sys.a.matmul(@p).matmul(@sys.a.transpose) + @q
    end

    def update(y : Float64Tensor)
      # Measurement residual: y_tilde = y - C * x
      y_tilde = y - @sys.c.matmul(@x)
      
      # Innovation covariance: S = C * P * C^T + R
      s = @sys.c.matmul(@p).matmul(@sys.c.transpose) + @r
      
      # Kalman Gain: K = P * C^T * S^-1
      k = @p.matmul(@sys.c.transpose).matmul(s.inv)
      
      # State update: x = x + K * y_tilde
      @x = @x + k.matmul(y_tilde)
      
      # Covariance update: P = (I - K * C) * P
      n = @sys.n_states
      eye = Float64Tensor.identity(n)
      @p = (eye - k.matmul(@sys.c)).matmul(@p)
    end
  end
end
