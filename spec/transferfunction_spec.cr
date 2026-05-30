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
end
