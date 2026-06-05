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

    it "performs pole placement using acker" do
      # Double integrator: A = [0 1; 0 0], B = [0; 1] (controllable)
      a = [[0.0, 1.0], [0.0, 0.0]].to_tensor
      b = [[0.0], [1.0]].to_tensor
      c = [[1.0, 0.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      # Place poles at -2, -3
      k = sys.acker([-2.0, -3.0])
      
      # For A - B*K, new characteristic eq should be s^2 + 5s + 6
      # K = [6, 5]
      k.to_unsafe[0].should be_close(6.0, 1e-5)
      k.to_unsafe[1].should be_close(5.0, 1e-5)
    end

    it "evaluates frequency response" do
      # G(s) = 1 / (s + 1)
      a = [[-1.0]].to_tensor
      b = [[1.0]].to_tensor
      c = [[1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      # Evaluate at omega = 1 rad/s
      # G(j1) = 1 / (j + 1) = 0.5 - 0.5j
      omega = [1.0].to_tensor
      res = sys.freqresp(omega)
      res.to_unsafe[0].real.should be_close(0.5, 1e-5)
      res.to_unsafe[0].imag.should be_close(-0.5, 1e-5)
    end

    it "solves Continuous Algebraic Riccati Equation (care)" do
      # Scalar system: A = 2, B = 1, Q = 3, R = 1
      # CARE: 2*A*P - P*B*R^-1*B^T*P + Q = 0  =>  4P - P^2 + 3 = 0  =>  P^2 - 4P - 3 = 0
      # Stable solution: P = 2 + sqrt(7) = 4.64575
      a = [[2.0]].to_tensor
      b = [[1.0]].to_tensor
      c = [[1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      q = [[3.0]].to_tensor
      r = [[1.0]].to_tensor
      p = sys.care(q, r)
      p.to_unsafe[0].should be_close(4.64575, 1e-5)
    end

    it "solves continuous-time Linear Quadratic Regulator (lqr)" do
      a = [[2.0]].to_tensor
      b = [[1.0]].to_tensor
      c = [[1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      q = [[3.0]].to_tensor
      r = [[1.0]].to_tensor
      k, p, poles = sys.lqr(q, r)
      p.to_unsafe[0].should be_close(4.64575, 1e-5)
      k.to_unsafe[0].should be_close(4.64575, 1e-5)
      poles[0].real.should be_close(-2.64575, 1e-5)
    end

    it "solves continuous-time Lyapunov equation (lyap)" do
      sys = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      p = sys.lyap([[1.0]].to_tensor)
      p.to_unsafe[0].should be_close(0.25, 1e-9)
    end

    it "solves discrete-time Lyapunov equation (dlyap)" do
      sys = CrySpace::StateSpace.new([[0.5]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      p = sys.dlyap([[1.0]].to_tensor)
      p.to_unsafe[0].should be_close(1.33333, 1e-5)
    end

    it "calculates stability margins" do
      num = [4.0].to_tensor
      den = [1.0, 2.0, 2.0].to_tensor
      sys = CrySpace::TransferFunction.new(num, den).to_statespace
      
      gm, gm_db, pm, w_gc, w_pc = sys.stability_margins
      gm.should eq(Float64::INFINITY)
      gm_db.should eq(Float64::INFINITY)
      pm.should be_close(68.54, 0.5)
      w_gc.should be_close(1.8622, 0.1)
      w_pc.should eq(-1.0)
    end

    it "solves Discrete Algebraic Riccati Equation (dare)" do
      a = [[2.0]].to_tensor
      b = [[1.0]].to_tensor
      c = [[1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d, 1.0)
      
      p = sys.dare([[3.0]].to_tensor, [[1.0]].to_tensor)
      p.to_unsafe[0].should be_close(6.4641, 1e-4)
    end

    it "solves discrete-time Linear Quadratic Regulator (dlqr)" do
      a = [[2.0]].to_tensor
      b = [[1.0]].to_tensor
      c = [[1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d, 1.0)
      
      k, p, poles = sys.dlqr([[3.0]].to_tensor, [[1.0]].to_tensor)
      p.to_unsafe[0].should be_close(6.4641, 1e-4)
      k.to_unsafe[0].should be_close(1.73205, 1e-4)
      poles[0].real.should be_close(0.26795, 1e-4)
    end

    it "performs observer pole placement using acker_obs" do
      a = [[0.0, 1.0], [0.0, 0.0]].to_tensor
      b = [[0.0], [1.0]].to_tensor
      c = [[1.0, 0.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      l = sys.acker_obs([-4.0, -5.0])
      l[0, 0].value.should be_close(9.0, 1e-5)
      l[1, 0].value.should be_close(20.0, 1e-5)
    end

    it "calculates controllability and observability Gramians" do
      sys = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[3.0]].to_tensor, [[0.0]].to_tensor)
      
      wc = sys.gram(:c)
      wc.to_unsafe[0].should be_close(0.25, 1e-6)
      
      wo = sys.gram(:o)
      wo.to_unsafe[0].should be_close(2.25, 1e-6)
    end

    it "calculates Hankel Singular Values (hsvd)" do
      sys = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[3.0]].to_tensor, [[0.0]].to_tensor)
      hsv = sys.hsvd
      hsv[0].should be_close(0.75, 1e-6)
    end

    it "runs linear simulation using lsim" do
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      t = Float64Tensor.linear_space(0.0, 2.0, 201)
      u = Float64Tensor.ones([1, t.size])
      
      t_out, x_out, y_out = sys.lsim(u, t)
      t_out.size.should eq(201)
      y_out[-1, 0].value.should be_close(1 - Math.exp(-2.0), 1e-4)
    end
end
