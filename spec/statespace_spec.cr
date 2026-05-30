require "./spec_helper"

describe CrySpace::StateSpace do
  it "initializes correctly" do
    a = [[1.0, 0.0], [0.0, 1.0]].to_tensor
    b = [[1.0], [0.0]].to_tensor
    c = [[1.0, 1.0]].to_tensor
    d = [[0.0]].to_tensor

    sys = CrySpace::StateSpace.new(a, b, c, d)
    sys.n_states.should eq(2)
    sys.n_inputs.should eq(1)
    sys.n_outputs.should eq(1)
  end

  it "calculates poles" do
    a = [[-1.0, 0.0], [0.0, -2.0]].to_tensor
    b = [[1.0], [1.0]].to_tensor
    c = [[1.0, 1.0]].to_tensor
    d = [[0.0]].to_tensor

    sys = CrySpace::StateSpace.new(a, b, c, d)
    poles = sys.poles
    # Sort complex by real part, then imaginary
    poles.sort_by! { |p| {p.real, p.imag} }
    poles[0].real.should eq(-2.0)
    poles[1].real.should eq(-1.0)
    end


    it "calculates dcgain" do
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor

    sys = CrySpace::StateSpace.new(a, b, c, d)
    gain = sys.dcgain
    gain[0, 0].should eq(1.0)
    end

    it "performs parallel connection (+)" do
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor

    sys = CrySpace::StateSpace.new(a, b, c, d)
    sys_sum = sys + sys

    sys_sum.n_states.should eq(2)
    sys_sum.dcgain[0, 0].should eq(2.0)
    end

    it "performs series connection (*)" do
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor

    sys1 = CrySpace::StateSpace.new(a, b, c, d)

    a2 = [[-2.0]].to_tensor
    b2 = [[1.0]].to_tensor
    c2 = [[2.0]].to_tensor
    d2 = [[0.0]].to_tensor
    sys2 = CrySpace::StateSpace.new(a2, b2, c2, d2)

    # sys_mul = sys1 * sys2 (sys2 then sys1)
    # sys2 gain = 2/2 = 1.0
    # sys1 gain = 1/1 = 1.0
    sys_mul = sys1 * sys2
    sys_mul.n_states.should eq(2)
    sys_mul.dcgain[0, 0].should eq(1.0)
    end

    it "performs feedback connection" do
    # G = 1/(s+1) => A=[-1], B=[1], C=[1], D=[0]
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d)

    # Feedback gain K=1
    k = [[1.0]].to_tensor
    sys_cl = sys.feedback(k)

    # Closed loop: 1/(s+2) => A=[-2], poles should be [-2.0]
    sys_cl.poles[0].should eq(-2.0)
    end

    it "discretizes system (sample)" do
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d)

    sys_d = sys.sample(1.0)
    sys_d.dt.should eq(1.0)
    sys_d.a[0, 0].value.should be_close(0.367879, 1e-5)
    # Bd = (1 - e^-1) * 1 = 0.63212
    sys_d.b[0, 0].value.should be_close(0.63212, 1e-5)
    end

    it "calculates step response" do
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d)

    t, x, y = sys.step_response(10)
    t.size.should eq(10)
    x.size.should eq(10)
    y.size.should eq(10)
    # After 9 steps of 0.1s (0.9s total), y = 1 - e^-0.9 = 0.593
    y.last[0, 0].value.should be_close(0.593, 0.01)
    # Also check state
    x.last[0, 0].value.should be_close(0.593, 0.01)
    end

    it "checks stability" do
    # Stable system
    sys_stable = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
    sys_stable.is_stable?.should be_true

    # Unstable system
    sys_unstable = CrySpace::StateSpace.new([[1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
    sys_unstable.is_stable?.should be_false
    end

    it "calculates controllability and observability" do
    a = [[0.0, 1.0], [-2.0, -3.0]].to_tensor
    b = [[0.0], [1.0]].to_tensor
    c = [[1.0, 0.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d)

    sys.is_controllable?.should be_true
    sys.is_observable?.should be_true

    # Unobservable system
    c2 = [[0.0, 0.0]].to_tensor
    sys2 = CrySpace::StateSpace.new(a, b, c2, d)
    sys2.is_observable?.should be_false
    end

    it "converts statespace to transferfunction" do
    # G(s) = 1 / (s + 1)
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d)

    tf = sys.to_transferfunction
    tf.num[0].value.should be_close(1.0, 1e-9)
    tf.den[0].value.should be_close(1.0, 1e-9)
    tf.den[1].value.should be_close(1.0, 1e-9)
    end

    it "calculates impulse response" do
    a = [[-1.0]].to_tensor
    b = [[1.0]].to_tensor
    c = [[1.0]].to_tensor
    d = [[0.0]].to_tensor
    sys = CrySpace::StateSpace.new(a, b, c, d)

    t, x, y = sys.impulse_response(10)
    # y(t) = e^-t. Since D=0, y[0]=0, y[1] should be approx C*B = 1.0
    y[0][0, 0].value.should eq(0.0)
    y[1][0, 0].value.should be_close(1.0, 0.1)
    y.last[0, 0].value.should be_close(Math.exp(-0.9), 0.1)
    end

    end



