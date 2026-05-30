require "./spec_helper"

describe CrySpace::Solver do
  it "solves using Euler" do
    # dx/dt = -x => x(t) = e^-t
    f = ->(x : Float64Tensor, t : Float64) {
      x * -1.0
    }
    x0 = [[1.0]].to_tensor
    t_span = {0.0, 1.0}
    dt = 0.01

    times, states = CrySpace::Solver.euler(f, x0, t_span, dt)
    
    times.last.should be_close(1.0, 1e-9)
    # Euler error is O(dt), for dt=0.01 it's about 1%
    states.last[0, 0].value.should be_close(Math.exp(-1.0), 0.01)
  end

  it "solves using RK4" do
    # dx/dt = -x => x(t) = e^-t
    f = ->(x : Float64Tensor, t : Float64) {
      x * -1.0
    }
    x0 = [[1.0]].to_tensor
    t_span = {0.0, 1.0}
    dt = 0.1 # Large step but RK4 is accurate

    times, states = CrySpace::Solver.rk4(f, x0, t_span, dt)
    
    times.last.should be_close(1.0, 1e-9)
    # RK4 error is O(dt^4), very accurate
    states.last[0, 0].value.should be_close(Math.exp(-1.0), 1e-5)
  end

  it "simulates StateSpace using RK4" do
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d)

    # Initial state 1.0, zero input
    t, x, y = sys.simulate({0.0, 1.0}, 0.1, x0: [[1.0]].to_tensor)
    
    x.last[0, 0].value.should be_close(Math.exp(-1.0), 1e-5)
    y.last[0, 0].value.should be_close(Math.exp(-1.0), 1e-5)
  end
end
