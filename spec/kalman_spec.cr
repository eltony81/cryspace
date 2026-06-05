require "./spec_helper"

describe CrySpace::KalmanFilter do
  it "updates state and covariance correctly" do
    # Simple discrete-time 1D random walk with drift
    dt = 1.0
    a = [[1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d, dt)

    q = [[0.1]].to_tensor
    r = [[0.5]].to_tensor

    x0 = [[0.0]].to_tensor
    p0 = [[1.0]].to_tensor

    kf = CrySpace::KalmanFilter.new(sys, q, r, x0, p0)

    # 1. Predict step
    # x_pred = 1 * 0 + 1 * 1 = 1.0
    # P_pred = 1 * 1 * 1 + 0.1 = 1.1
    kf.predict([[1.0]].to_tensor)
    kf.x[0, 0].value.should eq(1.0)
    kf.p[0, 0].value.should be_close(1.1, 1e-9)

    # 2. Update step
    # y = 1.5
    # y_tilde = 1.5 - 1.0 = 0.5
    # S = 1 * 1.1 * 1 + 0.5 = 1.6
    # K = 1.1 * 1 * (1/1.6) = 0.6875
    # x_updated = 1.0 + 0.6875 * 0.5 = 1.34375
    # P_updated = (1 - 0.6875 * 1) * 1.1 = 0.34375
    kf.update([[1.5]].to_tensor)
    kf.x[0, 0].value.should be_close(1.34375, 1e-5)
    kf.p[0, 0].value.should be_close(0.34375, 1e-5)
  end
end
