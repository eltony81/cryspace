require "./spec_helper"

describe CrySpace::PIDController do
  it "updates and saturates correctly with anti-windup" do
    # PID gains: Kp=2.0, Ki=5.0, Kd=0.5
    # tf = 0.01, saturation: min=-5.0, max=5.0
    pid = CrySpace::PIDController.new(2.0, 5.0, 0.5, 0.01, -5.0, 5.0)

    # First update: error = 1.0, dt = 0.1
    # P_term = 2.0 * 1.0 = 2.0
    # I_term_next = 0.0 + 5.0 * 1.0 * 0.1 = 0.5
    # D_term = (0.5 * 1.0 + 0) / (0.01 + 0.1) = 0.5 / 0.11 = 4.54545
    # Unsaturated U = 2.0 + 0.5 + 4.54545 = 7.04545
    # Saturated U = 5.0
    # Since saturated and error/unsat output have the same sign (error=1.0 > 0, unsat=7.04 > 0),
    # clamping anti-windup triggers and integral should freeze at 0.0.
    u = pid.update(1.0, 0.1)
    u.should eq(5.0)
    pid.integral.should eq(0.0)

    # Next update: error = -0.5, dt = 0.1 (error changes sign, should not clamp)
    # P_term = 2.0 * -0.5 = -1.0
    # prev_error = 1.0, error = -0.5
    # D_term = (0.5 * (-0.5 - 1.0) + 0.01 * 4.54545) / 0.11 = (-0.75 + 0.04545) / 0.11 = -6.4049
    # I_term_next = 0.0 + 5.0 * -0.5 * 0.1 = -0.25
    # Unsat U = -1.0 - 0.25 - 6.4049 = -7.6549
    # Saturated U = -5.0
    # Since error = -0.5 and unsat = -7.65 have same sign, clamping freezes integral at 0.0.
    u2 = pid.update(-0.5, 0.1)
    u2.should eq(-5.0)
    pid.integral.should eq(0.0)
  end
end
