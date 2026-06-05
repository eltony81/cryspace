require "./spec_helper"

describe CrySpace::Tuning do
  it "calculates Ziegler-Nichols tuning correctly" do
    # For G(s) = (1.5 * e^(-0.2 * s)) / (3.0 * s + 1)
    # K = 1.5, tau = 3.0, theta = 0.2
    # kp = (1.2 * 3.0) / (1.5 * 0.2) = 3.6 / 0.3 = 12.0
    # ti = 2 * 0.2 = 0.4  =>  ki = 12.0 / 0.4 = 30.0
    # td = 0.5 * 0.2 = 0.1  =>  kd = 12.0 * 0.1 = 1.2
    kp, ki, kd = CrySpace::Tuning.ziegler_nichols_fopdt(1.5, 3.0, 0.2)
    kp.should be_close(12.0, 1e-9)
    ki.should be_close(30.0, 1e-9)
    kd.should be_close(1.2, 1e-9)
  end

  it "raises on invalid dead time" do
    expect_raises(ArgumentError) do
      CrySpace::Tuning.ziegler_nichols_fopdt(1.0, 1.0, -0.5)
    end
  end
end
