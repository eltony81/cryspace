require "num"

module CrySpace
  class StateSpace
    def feedback(other : StateSpace, sign = -1)
      # Closed loop system with feedback
      n1 = n_states
      n2 = other.n_states
      
      # E = inv(I + D2 * D1)
      eye_outputs = Float64Tensor.identity(other.n_inputs)
      e = (eye_outputs + other.d.matmul(@d)).inv
      
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      a_cl[0...n1, 0...n1] = @a - @b.matmul(e).matmul(other.d).matmul(@c)
      a_cl[0...n1, n1...] = -@b.matmul(e).matmul(other.c)
      a_cl[n1..., 0...n1] = other.b.matmul(@c - @d.matmul(e).matmul(other.d).matmul(@c))
      a_cl[n1..., n1...] = other.a - other.b.matmul(@d).matmul(e).matmul(other.c)
      
      b_cl = Float64Tensor.zeros([n1 + n2, n_inputs])
      b_cl[0...n1, 0...] = @b.matmul(e)
      b_cl[n1..., 0...] = other.b.matmul(@d).matmul(e)
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      c_cl[0..., 0...n1] = @c - @d.matmul(e).matmul(other.d).matmul(@c)
      c_cl[0..., n1...] = -@d.matmul(e).matmul(other.c)
      
      d_cl = @d.matmul(e)
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
    end

    def feedback(k : Float64Tensor, sign = -1)
      # feedback with static gain K
      eye_k = Float64Tensor.identity(k.shape[0])
      e = (eye_k + k.matmul(@d)).inv
      
      a_cl = @a - @b.matmul(e).matmul(k).matmul(@c)
      b_cl = @b.matmul(e)
      
      eye_d = Float64Tensor.identity(n_outputs)
      inv_idk = (eye_d + @d.matmul(k)).inv
      
      c_cl = inv_idk.matmul(@c)
      d_cl = inv_idk.matmul(@d)
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
    end

    def +(other : StateSpace)
      unless n_inputs == other.n_inputs && n_outputs == other.n_outputs
        raise ArgumentError.new("Systems must have same number of inputs and outputs for parallel connection")
      end

      n1 = n_states
      n2 = other.n_states
      
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      n1.times { |r| n1.times { |c| a_cl[r, c] = @a[r, c].value } }
      n2.times { |r| n2.times { |c| a_cl[n1+r, n1+c] = other.a[r, c].value } }
      
      b_cl = Float64Tensor.zeros([n1 + n2, n_inputs])
      n1.times { |r| n_inputs.times { |c| b_cl[r, c] = @b[r, c].value } }
      n2.times { |r| n_inputs.times { |c| b_cl[n1+r, c] = other.b[r, c].value } }
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      n_outputs.times { |r| n1.times { |c| c_cl[r, c] = @c[r, c].value } }
      n_outputs.times { |r| n2.times { |c| c_cl[r, n1+c] = other.c[r, c].value } }
      
      d_cl = @d + other.d
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
    end

    def *(other : StateSpace)
      unless n_inputs == other.n_outputs
        raise ArgumentError.new("System 1 inputs must match System 2 outputs for series connection")
      end

      n1 = n_states
      n2 = other.n_states
      
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      bc = @b.matmul(other.c)
      n1.times { |r| n1.times { |c| a_cl[r, c] = @a[r, c].value } }
      n1.times { |r| n2.times { |c| a_cl[r, n1+c] = bc[r, c].value } }
      n2.times { |r| n2.times { |c| a_cl[n1+r, n1+c] = other.a[r, c].value } }
      
      b_cl = Float64Tensor.zeros([n1 + n2, other.n_inputs])
      bd = @b.matmul(other.d)
      n1.times { |r| other.n_inputs.times { |c| b_cl[r, c] = bd[r, c].value } }
      n2.times { |r| other.n_inputs.times { |c| b_cl[n1+r, c] = other.b[r, c].value } }
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      dc = @d.matmul(other.c)
      n_outputs.times { |r| n1.times { |c| c_cl[r, c] = @c[r, c].value } }
      n_outputs.times { |r| n2.times { |c| c_cl[r, n1+c] = dc[r, c].value } }
      
      d_cl = @d.matmul(other.d)
      
      StateSpace.new(a_cl, b_cl, c_cl, d_cl, @dt)
    end
  end

  def self.series(sys1 : StateSpace, sys2 : StateSpace) : StateSpace
    sys1 * sys2
  end

  def self.series(sys1 : TransferFunction, sys2 : TransferFunction) : TransferFunction
    sys1 * sys2
  end

  def self.parallel(sys1 : StateSpace, sys2 : StateSpace) : StateSpace
    sys1 + sys2
  end

  def self.parallel(sys1 : TransferFunction, sys2 : TransferFunction) : TransferFunction
    sys1 + sys2
  end

  def self.feedback(sys1 : StateSpace, sys2 : StateSpace, sign = -1) : StateSpace
    sys1.feedback(sys2, sign)
  end

  def self.feedback(sys1 : TransferFunction, sys2 : TransferFunction, sign = -1) : TransferFunction
    sys1.feedback(sys2, sign)
  end

  def self.append(sys1 : StateSpace, sys2 : StateSpace) : StateSpace
    n1 = sys1.n_states
    n2 = sys2.n_states
    m1 = sys1.n_inputs
    m2 = sys2.n_inputs
    p1 = sys1.n_outputs
    p2 = sys2.n_outputs
    
    a_new = Float64Tensor.zeros([n1 + n2, n1 + n2])
    a_new[0...n1, 0...n1] = sys1.a
    a_new[n1..., n1...] = sys2.a
    
    b_new = Float64Tensor.zeros([n1 + n2, m1 + m2])
    b_new[0...n1, 0...m1] = sys1.b
    b_new[n1..., m1...] = sys2.b
    
    c_new = Float64Tensor.zeros([p1 + p2, n1 + n2])
    c_new[0...p1, 0...n1] = sys1.c
    c_new[p1..., n1...] = sys2.c
    
    d_new = Float64Tensor.zeros([p1 + p2, m1 + m2])
    d_new[0...p1, 0...m1] = sys1.d
    d_new[p1..., m1...] = sys2.d
    
    dt = sys1.dt == sys2.dt ? sys1.dt : nil
    StateSpace.new(a_new, b_new, c_new, d_new, dt)
  end
end
