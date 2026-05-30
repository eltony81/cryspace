require "./spec_helper"

describe "CrySpace Complex Poles" do
  it "calculates complex poles for an oscillator" do
    # G(s) = 1 / (s^2 + 0.5s + 10)
    # Poles: -0.25 +/- 3.152j
    num = [1.0].to_tensor
    den = [1.0, 0.5, 10.0].to_tensor
    tf = CrySpace::TransferFunction.new(num, den)
    
    poles = tf.poles
    poles.size.should eq(2)
    poles.sort_by! { |p| p.imag }
    
    poles[0].real.should be_close(-0.25, 1e-6)
    poles[0].imag.should be_close(-3.15238, 1e-4)
    
    poles[1].real.should be_close(-0.25, 1e-6)
    poles[1].imag.should be_close(3.15238, 1e-4)
  end
end
