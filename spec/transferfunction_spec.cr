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
    tf.poles.to_a.sort.should eq([-2.0, -1.0])
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
end
