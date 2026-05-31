require "./spec_helper"

describe "CrySpace::StateSpace#feedback" do
  it "correctly computes the feedback loop for a simple gain system" do
    # Plant: G(s) = 1/(s+1) => A=[-1], B=[1], C=[1], D=[0]
    plant = CrySpace::StateSpace.new(
      [[-1.0]].to_tensor,
      [[1.0]].to_tensor,
      [[1.0]].to_tensor,
      [[0.0]].to_tensor
    )
    
    # Controller: Gain 1.0 => A=[], B=[], C=[], D=[1]
    # For a simple gain, we can model it as a StateSpace with 0 states
    # Note: CrySpace expects matrices, let's use a 1x1 A matrix with epsilon if needed, 
    # but let's try identity gain if possible.
    controller = CrySpace::StateSpace.new(
      [[0.0]].to_tensor,
      [[0.0]].to_tensor,
      [[0.0]].to_tensor,
      [[1.0]].to_tensor
    )
    
    # Wait, simple gain feedback loop is T = G / (1 + GH)
    # If G = 1/(s+1) and H=1, T = 1/(s+2)
    
    sys_cl = plant.feedback(controller, sign: -1)
    
    # Expected A for 1/(s+2) is [-2]
    # In our feedback implementation, A_cl will be [A - B*inv(I+D2*D1)*D2*C]
    # A_cl = -1 - (1 * inv(1 + 1*0) * 1 * 1) = -1 - 1 = -2
    
    sys_cl.a[0,0].value.should be_close(-2.0, 1e-6)
    
    # Verify DC gain
    sys_cl.dcgain[0,0].value.should be_close(0.5, 1e-6)
  end
end
