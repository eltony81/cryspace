require "num"

module CrySpace
  class UnscentedKalmanFilter
    property x : Float64Tensor
    property p : Float64Tensor
    
    def initialize(
      @f : Proc(Float64Tensor, Float64Tensor?, Float64Tensor),
      @h : Proc(Float64Tensor, Float64Tensor),
      @q : Float64Tensor,
      @r : Float64Tensor,
      x0 : Float64Tensor,
      p0 : Float64Tensor,
      @alpha = 1e-3,
      @beta = 2.0,
      @kappa = 0.0
    )
      @x = x0.dup
      @p = p0.dup
    end
    
    private def matrix_sqrt(p_mat : Float64Tensor) : Float64Tensor
      u, s, _ = p_mat.svd
      n = p_mat.shape[0]
      s_sqrt = Float64Tensor.zeros([n, n])
      n.times do |i|
        s_sqrt[i, i] = Math.sqrt([s[i].value, 0.0].max)
      end
      u.matmul(s_sqrt).matmul(u.transpose)
    end
    
    private def generate_sigma_points(mean : Float64Tensor, cov : Float64Tensor, lambda_val : Float64, l : Int32) : Array(Float64Tensor)
      # sigma_points: size 2*L + 1
      pts = Array(Float64Tensor).new(2 * l + 1)
      pts << mean.dup
      
      # sqrtm((L + lambda) * P)
      scaled_cov = cov * (l + lambda_val)
      sqrt_cov = matrix_sqrt(scaled_cov)
      
      l.times do |i|
        col = Float64Tensor.zeros([l, 1])
        l.times { |row| col[row, 0] = sqrt_cov[row, i].value }
        pts << (mean + col)
      end
      
      l.times do |i|
        col = Float64Tensor.zeros([l, 1])
        l.times { |row| col[row, 0] = sqrt_cov[row, i].value }
        pts << (mean - col)
      end
      pts
    end
    
    def predict(u : Float64Tensor? = nil)
      l = @x.shape[0]
      lambda_val = @alpha * @alpha * (l + @kappa) - l
      
      # 1. Weights
      w_m = Array(Float64).new(2 * l + 1, 0.0)
      w_c = Array(Float64).new(2 * l + 1, 0.0)
      w_m[0] = lambda_val / (l + lambda_val)
      w_c[0] = w_m[0] + (1.0 - @alpha * @alpha + @beta)
      
      (1..2*l).each do |i|
        w_m[i] = 1.0 / (2.0 * (l + lambda_val))
        w_c[i] = w_m[i]
      end
      
      # 2. Generate sigma points
      pts = generate_sigma_points(@x, @p, lambda_val, l)
      
      # 3. Propagate sigma points through f
      propagated = pts.map { |pt| @f.call(pt, u) }
      
      # 4. Predict mean state
      x_pred = Float64Tensor.zeros([l, 1])
      (2 * l + 1).times do |i|
        x_pred = x_pred + propagated[i] * w_m[i]
      end
      @x = x_pred
      
      # 5. Predict covariance P
      p_pred = Float64Tensor.zeros([l, l])
      (2 * l + 1).times do |i|
        diff = propagated[i] - @x
        p_pred = p_pred + diff.matmul(diff.transpose) * w_c[i]
      end
      @p = p_pred + @q
    end
    
    def update(y : Float64Tensor)
      l = @x.shape[0]
      ny = y.shape[0]
      lambda_val = @alpha * @alpha * (l + @kappa) - l
      
      # Weights
      w_m = Array(Float64).new(2 * l + 1, 0.0)
      w_c = Array(Float64).new(2 * l + 1, 0.0)
      w_m[0] = lambda_val / (l + lambda_val)
      w_c[0] = w_m[0] + (1.0 - @alpha * @alpha + @beta)
      
      (1..2*l).each do |i|
        w_m[i] = 1.0 / (2.0 * (l + lambda_val))
        w_c[i] = w_m[i]
      end
      
      # 1. Generate sigma points from predicted state
      pts = generate_sigma_points(@x, @p, lambda_val, l)
      
      # 2. Propagate through observation function h
      y_pts = pts.map { |pt| @h.call(pt) }
      
      # 3. Predict observation mean
      y_mean = Float64Tensor.zeros([ny, 1])
      (2 * l + 1).times do |i|
        y_mean = y_mean + y_pts[i] * w_m[i]
      end
      
      # 4. Compute innovation covariance Pyy and cross-covariance Pxy
      pyy = Float64Tensor.zeros([ny, ny])
      pxy = Float64Tensor.zeros([l, ny])
      
      (2 * l + 1).times do |i|
        y_diff = y_pts[i] - y_mean
        x_diff = pts[i] - @x
        
        pyy = pyy + y_diff.matmul(y_diff.transpose) * w_c[i]
        pxy = pxy + x_diff.matmul(y_diff.transpose) * w_c[i]
      end
      
      pyy = pyy + @r
      
      # 5. Kalman gain: K = Pxy * Pyy^-1
      k = pxy.matmul(pyy.inv)
      
      # 6. Update state and covariance
      @x = @x + k.matmul(y - y_mean)
      @p = @p - k.matmul(pyy).matmul(k.transpose)
    end
  end
end
