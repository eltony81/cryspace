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

    it "converts to observability canonical form" do
      num = [1.0].to_tensor
      den = [1.0, 1.0].to_tensor
      sys = CrySpace::TransferFunction.new(num, den).to_statespace
      
      sys_o = sys.to_observability_form
      sys_o.n_states.should eq(1)
      sys_o.a[0, 0].value.should eq(-1.0)
    end

    it "converts to modal canonical form" do
      a = [[-1.0, 0.0], [0.0, -2.0]].to_tensor
      b = [[1.0], [1.0]].to_tensor
      c = [[1.0, 1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      sys_m = sys.to_modal_form
      poles = sys_m.poles.sort_by { |p| {p.real, p.imag} }
      poles[0].real.should be_close(-2.0, 1e-4)
      poles[1].real.should be_close(-1.0, 1e-4)
    end

    it "evaluates root locus gains" do
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      gains = [0.0, 1.0, 2.0].to_tensor
      poles_arr = sys.root_locus(gains)
      
      poles_arr[0][0].real.should be_close(-1.0, 1e-9)
      poles_arr[1][0].real.should be_close(-2.0, 1e-9)
      poles_arr[2][0].real.should be_close(-3.0, 1e-9)
    end

    it "computes bode_data and nyquist_data" do
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      omega = [1.0].to_tensor
      
      _, db, deg = sys.bode_data(omega)
      db[0].should be_close(-3.0103, 1e-3)
      deg[0].should be_close(-45.0, 1e-3)
      
      real_parts, imag_parts = sys.nyquist_data(omega)
      real_parts[0].should be_close(0.5, 1e-5)
      imag_parts[0].should be_close(-0.5, 1e-5)
    end

    it "performs discrete-to-continuous conversion (to_continuous)" do
      a = [[-2.0]].to_tensor
      b = [[1.0]].to_tensor
      c = [[1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys_c = CrySpace::StateSpace.new(a, b, c, d)
      sys_d = sys_c.sample(0.5)
      
      sys_back = sys_d.to_continuous
      sys_back.a[0, 0].value.should be_close(-2.0, 1e-4)
      sys_back.b[0, 0].value.should be_close(1.0, 1e-4)
    end

    it "performs Bilinear/Tustin discretization" do
      # Continuous: G(s) = 1 / (s + 2) => A=-2, B=1, C=1, D=0
      sys_c = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      dt = 0.1
      # Tustin (alpha=0.5):
      # Ad = (1 - 0.05 * A)^-1 * (1 + 0.05 * A) = (1 - 0.05 * -2)^-1 * (1 + 0.05 * -2) = 1.1^-1 * 0.9 = 0.9 / 1.1 = 0.81818
      # Bd = (1 - 0.05 * A)^-1 * B * dt = 1.1^-1 * 1.0 * 0.1 = 0.1 / 1.1 = 0.090909
      sys_d = sys_c.sample(dt, method: :tustin)
      sys_d.a[0, 0].value.should be_close(0.81818, 1e-4)
      sys_d.b[0, 0].value.should be_close(0.090909, 1e-4)
    end

    it "performs balanced truncation model reduction (balred)" do
      # 2nd-order system with two decoupled stable states: A = [[-1, 0], [0, -10]], B=[[1], [1]], C=[[1, 1]]
      # State 2 (pole at -10) is much faster and less dominant than State 1 (pole at -1)
      a = [[-1.0, 0.0], [0.0, -10.0]].to_tensor
      b = [[1.0], [1.0]].to_tensor
      c = [[1.0, 1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      sys_reduced = sys.balred(1)
      sys_reduced.n_states.should eq(1)
      # The dominant pole is at -1.328 (from balanced truncation of poles at -1 and -10)
      sys_reduced.poles[0].real.should be_close(-1.328, 1e-3)
    end

    it "designs an LQG controller" do
      a = [[0.0, 1.0], [-2.0, -3.0]].to_tensor
      b = [[0.0], [1.0]].to_tensor
      c = [[1.0, 0.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      k_gain = [[1.0, 2.0]].to_tensor
      l_gain = [[3.0], [4.0]].to_tensor
      
      controller = sys.lqg(k_gain, l_gain)
      controller.n_states.should eq(2)
      controller.n_inputs.should eq(1)  # Takes y as input
      controller.n_outputs.should eq(1) # Produces u as output
    end

    it "computes nichols_data and margin" do
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      omega = [1.0].to_tensor
      
      _, db, deg = sys.nichols_data(omega)
      db[0].should be_close(-3.0103, 1e-3)
      deg[0].should be_close(-45.0, 1e-3)
      
      # Margin
      gm, gm_db, pm, w_gc, w_pc = sys.stability_margins
      gm.should eq(Float64::INFINITY)
      pm.should eq(0.0)
    end

    it "performs similarity transformations" do
      # System: dx/dt = -x + u, y = x
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      # Transform state: z = 2 * x => x = 0.5 * z
      t_matrix = [[2.0]].to_tensor
      sys_t = sys.similarity_transform(t_matrix)
      
      sys_t.a[0, 0].value.should be_close(-1.0, 1e-9)
      sys_t.b[0, 0].value.should be_close(0.5, 1e-9)
      sys_t.c[0, 0].value.should be_close(2.0, 1e-9)
    end

    it "performs Kalman controllability & observability decompositions" do
      # 2nd-order system where state 2 is uncontrollable: dx/dt = [[-1, 0], [0, -2]]*x + [[1], [0]]*u
      a = [[-1.0, 0.0], [0.0, -2.0]].to_tensor
      b = [[1.0], [0.0]].to_tensor
      c = [[1.0, 1.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      sys_c, _, r_c = sys.controllable_decomposition
      r_c.should eq(1) # Controllable rank is 1
      
      # 2nd-order system where state 2 is unobservable: y = [[1, 0]]*x
      c2 = [[1.0, 0.0]].to_tensor
      sys2 = CrySpace::StateSpace.new(a, b, c2, d)
      
      sys_o, _, r_o = sys2.observable_decomposition
      r_o.should eq(1) # Observable rank is 1
    end

    it "finds minimal realization using minreal" do
      # System has 3 states: State 1 (stable/controllable/observable),
      # State 2 (uncontrollable), State 3 (unobservable)
      # Should reduce from 3 states to 1 state
      a = [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -3.0]].to_tensor
      b = [[1.0], [0.0], [1.0]].to_tensor
      c = [[1.0, 1.0, 0.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)
      
      sys_min = sys.minreal
      sys_min.n_states.should eq(1)
      sys_min.poles[0].real.should be_close(-1.0, 1e-4)
    end

    it "augments systems with integral action" do
      # Double integrator system
      sys = CrySpace::StateSpace.new([[0.0, 1.0], [0.0, 0.0]].to_tensor, [[0.0], [1.0]].to_tensor, [[1.0, 0.0]].to_tensor, [[0.0]].to_tensor)
      sys_aug = sys.augment_integrator
      
      sys_aug.n_states.should eq(3) # 2 plant states + 1 integrator state
      sys_aug.n_inputs.should eq(1)
      sys_aug.n_outputs.should eq(1)
      
      sys_aug.a[2, 0].value.should be_close(-1.0, 1e-9) # -C row
    end

    it "calculates optimal Kalman estimator gains via LQE / DLQE" do
      # SISO System: G(s) = 1 / (s + 1)
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      
      q = [[1.0]].to_tensor
      r = [[1.0]].to_tensor
      
      # Continuous LQE:
      # P solves: -2*P - P^2 + 1 = 0 => P^2 + 2*P - 1 = 0 => P = -1 + sqrt(2) = 0.4142
      # L = P * C^T * R^-1 = 0.4142
      l_gain = sys.lqe(q, r)
      l_gain[0, 0].value.should be_close(0.4142, 1e-4)
      
      # Discrete DLQE:
      sys_d = sys.sample(0.5)
      l_gain_d = sys_d.dlqe(q, r)
      l_gain_d[0, 0].value.should_not be_nil
    end

    it "computes bandwidth and initial_response" do
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      
      # Bandwidth of 1/(s+1) should be 1.0 rad/s (-3dB point)
      w_bw = sys.bandwidth
      w_bw.should be_close(1.0, 0.1)

      # Initial response starting from x0 = 2.0
      x0 = [[2.0]].to_tensor
      t, x, y = sys.initial_response(x0, n_steps: 5)
      y[0][0, 0].value.should be_close(2.0, 1e-9)
      y.last[0, 0].value.should be_close(2.0 * Math.exp(-0.4), 0.1)
    end

    it "transforms to control/observable canonical forms" do
      # 2nd order system: G(s) = 1 / (s^2 + 3s + 2)
      a = [[0.0, 1.0], [-2.0, -3.0]].to_tensor
      b = [[0.0], [1.0]].to_tensor
      c = [[1.0, 0.0]].to_tensor
      d = [[0.0]].to_tensor
      sys = CrySpace::StateSpace.new(a, b, c, d)

      # Control Canonical Form
      sys_ccf, t_ccf = sys.to_control_canonical_form
      sys_ccf.n_states.should eq(2)
      # Observable Canonical Form
      sys_ocf, t_ocf = sys.to_observable_canonical_form
      sys_ocf.n_states.should eq(2)
    end

    it "performs feedback and minreal on TransferFunction" do
      # G(s) = 1 / (s + 2)
      num1 = [1.0].to_tensor
      den1 = [1.0, 2.0].to_tensor
      g1 = CrySpace::TransferFunction.new(num1, den1)

      # Feedback G1 / (1 + G1) => 1 / (s + 3)
      g_cl = g1.feedback(CrySpace::TransferFunction.new([1.0].to_tensor, [1.0].to_tensor))
      g_cl.den[1].value.should be_close(3.0, 1e-9)

      # Minreal: G(s) = (s + 1) / ( (s+1)(s+2) ) => 1 / (s + 2)
      num2 = [1.0, 1.0].to_tensor
      den2 = [1.0, 3.0, 2.0].to_tensor # s^2 + 3s + 2
      g2 = CrySpace::TransferFunction.new(num2, den2)
      g2_min = g2.minreal
      g2_min.poles.size.should eq(1)
      g2_min.poles[0].real.should be_close(-2.0, 1e-3)
    end

    it "performs Pade delay approximation" do
      # 1st-order Pade delay of 1.0s: e^-s ≈ (2 - s) / (2 + s) => (-s + 2) / (s + 2)
      g_delay = CrySpace::TransferFunction.pade(1.0, 1)
      g_delay.num[0].value.should be_close(-1.0, 1e-9)
      g_delay.num[1].value.should be_close(2.0, 1e-9)
      g_delay.den[1].value.should be_close(2.0, 1e-9)
    end

    it "computes ss2tf, peak_gain, and loop_margins" do
      # G(s) = 1 / (s + 1)
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      
      # ss2tf
      tf = sys.ss2tf
      tf.den[1].value.should be_close(1.0, 1e-9)
      
      # peak gain is DC gain of 1.0
      sys.peak_gain.should be_close(1.0, 1e-4)

      # loop margins: critical point distance is 1.0 at w=0 since 1+G(0)=2
      gm, pm = sys.loop_margins
      gm[0].should be_close(0.5, 0.1) # 1 / (1 + 1) = 0.5
      pm[1].should be_close(60.0, 5.0) # approx 60 degrees Phase Margin
    end

    it "performs H2 and H-infinity control synthesis" do
      # SISO Stable system
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      c_z = [[1.0]].to_tensor
      d_zu = [[1.0]].to_tensor

      # H2 synthesis (equivalent to LQR)
      k_h2, p_h2 = sys.h2syn(c_z, d_zu)
      k_h2[0, 0].value.should be_close(0.4142, 1e-4)

      # H-infinity synthesis at gamma = 2.0
      k_hinf, p_hinf = sys.hinfsyn(c_z, d_zu, gamma: 2.0)
      k_hinf[0, 0].value.should_not be_nil
    end

    it "computes stepinfo response metrics" do
      # SISO System: G(s) = 1 / (s + 2)
      # Step response: y(t) = 0.5 * (1 - e^(-2t))
      # Steady state value is 0.5.
      # Rise time (10% to 90%): 
      # 0.1 * 0.5 = 0.05 => 1 - e^-2t1 = 0.1 => t1 = -ln(0.9)/2 ≈ 0.0526
      # 0.9 * 0.5 = 0.45 => 1 - e^-2t2 = 0.9 => t2 = -ln(0.1)/2 ≈ 1.1513
      # Rise time = t2 - t1 ≈ 1.0987 s.
      sys = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      
      info = sys.stepinfo(n_steps: 300)
      info.steady_state_value.should be_close(0.5, 0.01)
      info.rise_time.should be_close(1.0987, 0.15)
      info.overshoot.should eq(0.0) # first-order system has no overshoot
      info.peak.should be_close(0.5, 0.05)
    end

    it "simulates closed-loop observer dynamics" do
      # System: G(s) = 1 / (s + 1)
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      
      # Gains: K = 1.0, L = 5.0
      k_gain = [[1.0]].to_tensor
      l_gain = [[5.0]].to_tensor
      
      t = Float64Tensor.linear_space(0.0, 5.0, 100)
      x0 = [[2.0]].to_tensor
      x0_est = [[0.0]].to_tensor # initial estimation error is 2.0
      
      # Observer simulation without noise
      t_out, x_out, x_est_out, y_out, u_out = sys.simulate_observer(t, k_gain, l_gain, x0: x0, x0_est: x0_est)
      
      # Estimated state should converge to true state
      final_error = (x_out[99, 0].value - x_est_out[99, 0].value).abs
      final_error.should be_close(0.0, 1e-3)
      
      # With noise
      q = [[0.01]].to_tensor
      r = [[0.01]].to_tensor
      t_out2, x_out2, x_est_out2, y_out2, u_out2 = sys.simulate_observer(t, k_gain, l_gain, x0: x0, x0_est: x0_est, process_noise_cov: q, measure_noise_cov: r)
      t_out2.size.should eq(100)
      x_out2.shape[0].should eq(100)
    end

    it "computes sigma_data singular values" do
      # SISO System: G(s) = 2 / (s + 3)
      sys = CrySpace::StateSpace.new([[-3.0]].to_tensor, [[2.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      omega = [0.0].to_tensor # DC frequency
      w, sv = sys.sigma_data(omega)
      # At w=0, G(0) = 2/3 ≈ 0.6667
      sv[0, 0].value.should be_close(2.0 / 3.0, 1e-4)
    end

    it "computes nbar prefilter tracking scaling gain" do
      # SISO System: G(s) = 2 / (s + 3)
      sys = CrySpace::StateSpace.new([[-3.0]].to_tensor, [[2.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      k_gain = [[1.0]].to_tensor # u = -Kx
      # A_cl = A - BK = -3 - 2*1 = -5
      # N = inv(-C * inv(A_cl) * B) = inv(-1 * inv(-5) * 2) = inv(2/5) = 2.5
      n_scale = sys.nbar(k_gain)
      n_scale[0, 0].value.should be_close(2.5, 1e-4)
    end

    it "simulates ramp_response" do
      # G(s) = 1/s => integrators step input yields ramp output, ramp input yields quadratic output
      sys = CrySpace::StateSpace.new([[0.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      t, x, y = sys.ramp_response(n_steps: 10)
      t[1].should be_close(0.1, 1e-9)
      y[0][0, 0].value.should eq(0.0)
    end

    it "performs CrySpace module level connection algebra" do
      sys1 = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      sys2 = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      
      sys_s = CrySpace.series(sys1, sys2)
      sys_s.n_states.should eq(2)
      
      sys_p = CrySpace.parallel(sys1, sys2)
      sys_p.n_states.should eq(2)
      
      sys_f = CrySpace.feedback(sys1, sys2)
      sys_f.n_states.should eq(2)

      sys_a = CrySpace.append(sys1, sys2)
      sys_a.n_states.should eq(2)
      sys_a.n_inputs.should eq(2)
      sys_a.n_outputs.should eq(2)
    end

    it "creates static gain and eye systems" do
      sys_eye = CrySpace::StateSpace.eye(2)
      sys_eye.n_states.should eq(0)
      sys_eye.n_inputs.should eq(2)
      sys_eye.n_outputs.should eq(2)
      sys_eye.d[0, 0].value.should eq(1.0)
    end

    it "computes transmission zeros" do
      # G(s) = (s + 4) / (s + 3) => zero = -4
      sys = CrySpace::StateSpace.new([[-3.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor)
      zeros = sys.transmission_zeros
      zeros.size.should eq(1)
      zeros[0].real.should be_close(-4.0, 1e-5)
    end

    it "computes balreal balanced realization" do
      sys = CrySpace::StateSpace.new([[-1.0, 0.0], [0.0, -2.0]].to_tensor, [[1.0], [1.0]].to_tensor, [[1.0, 1.0]].to_tensor, [[0.0]].to_tensor)
      sys_bal, t_mat, t_inv = sys.balreal
      sys_bal.n_states.should eq(2)
      # Controllability and observability gramians of the balanced system should be diagonal and equal
      sys_bal.gram(:c)[0, 0].value.should be_close(sys_bal.gram(:o)[0, 0].value, 1e-4)
    end

    it "computes sensitivity and complementary sensitivity" do
      sys = CrySpace::StateSpace.new([[-1.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      k = CrySpace::StateSpace.new([[-2.0]].to_tensor, [[1.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      
      sens = sys.sensitivity(k)
      sens.n_states.should eq(2)
      
      csens = sys.complementary_sensitivity(k)
      csens.n_states.should eq(2)
    end

    it "solves LQR and DLQR with cross-coupling cost matrix" do
      # dx/dt = -3*x + 2*u
      sys = CrySpace::StateSpace.new([[-3.0]].to_tensor, [[2.0]].to_tensor, [[1.0]].to_tensor, [[0.0]].to_tensor)
      q = [[1.0]].to_tensor
      r = [[1.0]].to_tensor
      n_cross = [[0.5]].to_tensor
      
      k, _, _ = sys.lqr(q, r, n_cross)
      k[0, 0].value.should_not be_nil

      sys_d = sys.sample(0.1)
      kd, _, _ = sys_d.dlqr(q, r, n_cross)
      kd[0, 0].value.should_not be_nil
    end
  end
