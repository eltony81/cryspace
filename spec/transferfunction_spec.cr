require "./spec_helper"

describe CrySpace::TransferFunction do
  it "initializes correctly" do
    num = [1.0].to_tensor
    den = [1.0, 1.0].to_tensor
    tf = CrySpace::TransferFunction.new(num, den)
    tf.num.size.should eq(1)
    tf.den.size.should eq(2)
  end

  it "calculates poles" do
    # G(s) = 1/(s^2 + 3s + 2) => poles = -1, -2
    num = [1.0].to_tensor
    den = [1.0, 3.0, 2.0].to_tensor
    tf = CrySpace::TransferFunction.new(num, den)
    poles = tf.poles
    poles.sort_by! { |p| {p.real, p.imag} }
    poles[0].real.should eq(-2.0)
    poles[1].real.should eq(-1.0)
  end

  it "converts to statespace" do
    # G(s) = 1/(s+1)
    num = [1.0].to_tensor
    den = [1.0, 1.0].to_tensor
    tf = CrySpace::TransferFunction.new(num, den)

    ss = tf.to_statespace
    ss.n_states.should eq(1)
    ss.a[0, 0].value.should eq(-1.0)
    ss.dcgain[0, 0].value.should eq(1.0)
  end

  it "calculates zeros" do
    # G(s) = (s + 2) / (s^2 + 3s + 2) => zero = -2
    num = [1.0, 2.0].to_tensor
    den = [1.0, 3.0, 2.0].to_tensor
    tf = CrySpace::TransferFunction.new(num, den)
    tf.zeros[0].real.should eq(-2.0)
  end

  it "performs transferfunction arithmetic" do
    num = [1.0].to_tensor
    den = [1.0, 1.0].to_tensor
    tf1 = CrySpace::TransferFunction.new(num, den) # 1/(s+1)
    
    # Series: 1/(s+1) * 1/(s+1) = 1/(s^2 + 2s + 1)
    tf_series = tf1 * tf1
    tf_series.den[1].value.should eq(2.0)
    tf_series.den[2].value.should eq(1.0)

    # Parallel: 1/(s+1) + 1/(s+1) = (2s + 2) / (s^2 + 2s + 1)
    tf_parallel = tf1 + tf1
    tf_parallel.num[0].value.should eq(2.0)
    tf_parallel.num[1].value.should eq(2.0)
    tf_parallel.den[1].value.should eq(2.0)
    tf_parallel.den[2].value.should eq(1.0)
  end

  it "creates lead and lag compensators" do
    lead = CrySpace::TransferFunction.lead_compensator(gain: 2.0, zero: 1.0, pole: 10.0)
    lead.num[0].value.should be_close(2.0, 1e-9)
    lead.num[1].value.should be_close(2.0, 1e-9)
    lead.den[1].value.should be_close(10.0, 1e-9)

    lag = CrySpace::TransferFunction.lag_compensator(gain: 3.0, zero: 5.0, pole: 0.5)
    lag.num[0].value.should be_close(3.0, 1e-9)
    lag.num[1].value.should be_close(15.0, 1e-9)
    lag.den[1].value.should be_close(0.5, 1e-9)
  end

  it "designs Butterworth filters" do
    # 1st-order Butterworth lowpass with cutoff Wn = 2.0 rad/s
    # G(s) = 2 / (s + 2)
    but1 = CrySpace::TransferFunction.butter(1, 2.0)
    but1.num[0].value.should be_close(2.0, 1e-9)
    but1.den[1].value.should be_close(2.0, 1e-9)

    # 2nd-order Butterworth lowpass with Wn = 1.0 rad/s
    # G(s) = 1 / (s^2 + sqrt(2)*s + 1)
    but2 = CrySpace::TransferFunction.butter(2, 1.0)
    but2.num[0].value.should be_close(1.0, 1e-9)
    but2.den[1].value.should be_close(Math.sqrt(2.0), 1e-5)
    but2.den[2].value.should be_close(1.0, 1e-9)
  end

  it "computes TF sensitivity and complementary sensitivity" do
    # G(s) = 1 / (s + 1)
    g = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 1.0].to_tensor)
    k = CrySpace::TransferFunction.new([2.0].to_tensor, [1.0].to_tensor)
    
    # S = 1 / (1 + G*K) = (s + 1) / (s + 3)
    s = g.sensitivity(k)
    s.den[1].value.should be_close(3.0, 1e-9)
    
    # T = G*K / (1 + G*K) = 2 / (s + 3)
    t = g.complementary_sensitivity(k)
    t.num[0].value.should be_close(2.0, 1e-9)
    t.den[1].value.should be_close(3.0, 1e-9)
  end

  it "performs filter frequency transformations" do
    # 1st-order Butterworth lowpass with Wn = 1.0 rad/s: G(s) = 1 / (s + 1)
    lp = CrySpace::TransferFunction.butter(1, 1.0)
    
    # Lowpass to Highpass with cutoff 3.0: G_hp(s) = s / (s + 3)
    hp = lp.lowpass_to_highpass(3.0)
    hp.num[0].value.should be_close(1.0, 1e-9)
    hp.den[1].value.should be_close(3.0, 1e-9)
    
    # Lowpass to Bandpass with center 2.0 and BW 0.5
    bp = lp.lowpass_to_bandpass(2.0, 0.5)
    bp.poles.size.should eq(2)
  end

  it "performs TF discretization and continuous conversion" do
    # G(s) = 1 / (s + 2)
    g = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 2.0].to_tensor)
    g_d = g.to_discrete(0.1, :zoh)
    g_d.dt.should eq(0.1)
    
    g_c = g_d.to_continuous
    g_c.den[1].value.should be_close(2.0, 1e-3)
  end

  it "designs Chebyshev Type I and digital Butterworth filters" do
    cheb = CrySpace::TransferFunction.cheby1(1, 1.0, 2.0)
    cheb.num.size.should eq(1)
    cheb.den.size.should eq(2)
    
    but_d = CrySpace::TransferFunction.butter_digital(1, 10.0, 0.05)
    but_d.dt.should eq(0.05)
  end

  it "discretizes using matched pole-zero method (:matched)" do
    # G(s) = 1 / (s + 2)
    # Pole s = -2 maps to z = e^(-2 * 0.1) = e^(-0.2) = 0.81873
    # With matched, continuous gain at s=0 is 0.5.
    # Discrete system should have pole at z = 0.81873 and match gain at DC:
    # G_d(z) = K_d / (z - 0.81873)  =>  G_d(1) = K_d / (1 - 0.81873) = 0.5  =>  K_d = 0.5 * 0.18127 = 0.090634
    g = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 2.0].to_tensor)
    g_d = g.to_discrete(0.1, :matched)
    g_d.den[1].value.should be_close(-0.81873, 1e-4)
    g_d.num[0].value.should be_close(0.090634, 1e-4)
  end

  it "designs lead-lag compensator (leadlag) and loop shaping weight (makeweight)" do
    comp = CrySpace::TransferFunction.leadlag(2.0, 10.0, 1.5)
    comp.num[0].value.should eq(1.5)
    comp.num[1].value.should eq(3.0)
    comp.den[1].value.should eq(10.0)
    
    w = CrySpace::TransferFunction.makeweight(0.1, 10.0, 2.0)
    w.num[0].value.should be_close(0.5, 1e-5) # 1 / 2.0
    w.num[1].value.should be_close(10.0, 1e-5)
    w.den[1].value.should be_close(1.0, 1e-5) # 10 * 0.1
  end
end
