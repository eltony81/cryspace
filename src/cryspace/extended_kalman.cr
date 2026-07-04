require "num"

module CrySpace
  class ExtendedKalmanFilter
    property x : Float64Tensor
    property p : Float64Tensor
    
    def initialize(
      @f : Proc(Float64Tensor, Float64Tensor?, Float64Tensor),
      @h : Proc(Float64Tensor, Float64Tensor),
      @jf : Proc(Float64Tensor, Float64Tensor?, Float64Tensor),
      @jh : Proc(Float64Tensor, Float64Tensor),
      @q : Float64Tensor,
      @r : Float64Tensor,
      x0 : Float64Tensor,
      p0 : Float64Tensor
    )
      @x = x0.dup
      @p = p0.dup
    end
    
    def predict(u : Float64Tensor? = nil)
      # Jacobian F evaluated at the prior estimate x_{k-1|k-1}
      f_jac = @jf.call(@x, u)

      # Predict state: x = f(x, u)
      @x = @f.call(@x, u)

      # P = F * P * F^T + Q
      @p = f_jac.matmul(@p).matmul(f_jac.transpose) + @q
    end
    
    def update(y : Float64Tensor)
      # Jacobian H
      h_jac = @jh.call(@x)
      
      # Innovation covariance: S = H * P * H^T + R
      s = h_jac.matmul(@p).matmul(h_jac.transpose) + @r
      
      # Kalman Gain: K = P * H^T * S^-1
      k = @p.matmul(h_jac.transpose).matmul(s.inv)
      
      # Innovation error: y - h(x)
      innov = y - @h.call(@x)
      
      # Update state: x = x + K * innov
      @x = @x + k.matmul(innov)
      
      # Update covariance: P = (I - K * H) * P
      n = @x.shape[0]
      eye = Float64Tensor.identity(n)
      @p = (eye - k.matmul(h_jac)).matmul(@p)
    end
  end
end
