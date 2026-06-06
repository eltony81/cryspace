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
end
