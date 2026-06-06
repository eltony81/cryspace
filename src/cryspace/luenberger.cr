require "num"

module CrySpace
  class LuenbergerObserver
    property sys : StateSpace
    property l : Float64Tensor
    property x_hat : Float64Tensor

    def initialize(@sys : StateSpace, @l : Float64Tensor, x0_hat : Float64Tensor? = nil)
      n = @sys.n_states
      p = @sys.n_outputs
      
      unless @l.shape == [n, p]
        raise ArgumentError.new("Observer gain matrix L must have shape [#{n}, #{p}], got #{@l.shape}")
      end

      if x0_hat
        unless x0_hat.shape == [n, 1] || x0_hat.shape == [n]
          raise ArgumentError.new("Initial state estimate x0_hat must have shape [#{n}, 1] or [#{n}], got #{x0_hat.shape}")
        end
        @x_hat = x0_hat.rank == 1 ? x0_hat.reshape([n, 1]) : x0_hat.dup
      else
        @x_hat = Float64Tensor.zeros([n, 1])
      end
    end

    # Discrete-time update:
    # x_hat_{k+1} = A * x_hat_k + B * u_k + L * (y_k - C * x_hat_k - D * u_k)
    def update_discrete(y : Float64Tensor, u : Float64Tensor) : Float64Tensor
      y_col = y.rank == 1 ? y.reshape([@sys.n_outputs, 1]) : y
      u_col = u.rank == 1 ? u.reshape([@sys.n_inputs, 1]) : u

      y_hat = @sys.c.matmul(@x_hat) + @sys.d.matmul(u_col)
      error = y_col - y_hat
      
      @x_hat = @sys.a.matmul(@x_hat) + @sys.b.matmul(u_col) + @l.matmul(error)
      @x_hat
    end

    # Continuous-time update using a single RK4 step of size dt:
    # dx_hat/dt = A * x_hat + B * u + L * (y - C * x_hat - D * u)
    def update_continuous(y : Float64Tensor, u : Float64Tensor, dt : Float64) : Float64Tensor
      y_col = y.rank == 1 ? y.reshape([@sys.n_outputs, 1]) : y
      u_col = u.rank == 1 ? u.reshape([@sys.n_inputs, 1]) : u

      f = ->(xh : Float64Tensor) {
        y_hat = @sys.c.matmul(xh) + @sys.d.matmul(u_col)
        error = y_col - y_hat
        @sys.a.matmul(xh) + @sys.b.matmul(u_col) + @l.matmul(error)
      }

      k1 = f.call(@x_hat)
      k2 = f.call(@x_hat + k1 * (dt / 2.0))
      k3 = f.call(@x_hat + k2 * (dt / 2.0))
      k4 = f.call(@x_hat + k3 * dt)

      @x_hat = @x_hat + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
      @x_hat
    end
  end
end
