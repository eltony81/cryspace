require "num"
module CrySpace
  class StateSpace
    getter a : Float64Tensor, b : Float64Tensor, c : Float64Tensor, d : Float64Tensor
    getter n_states : Int32, n_inputs : Int32, n_outputs : Int32
    def initialize(@a, @b, @c, @d)
      @n_states = @a.shape[0].to_i32
      @n_inputs = @b.shape[1].to_i32
      @n_outputs = @c.shape[0].to_i32
    end
    def dcgain; eye = Float64Tensor.identity(@n_states); @d - @c.matmul((eye - @a).solve(@b)); end
    def feedback(other : StateSpace, sign = -1)
      n1 = n_states; n2 = other.n_states
      eye_outputs = Float64Tensor.identity(other.n_inputs)
      e = (eye_outputs - (other.d.matmul(@d)) * sign).inv
      # G/(1 - sign*GH). Per negative feedback, sign=-1 => G/(1+GH)
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      a_cl[0...n1, 0...n1] = @a - @b.matmul(e).matmul(other.d).matmul(@c) * sign
      a_cl[0...n1, n1...] = -@b.matmul(e).matmul(other.c)
      a_cl[n1..., 0...n1] = other.b.matmul(@c - @d.matmul(e).matmul(other.d).matmul(@c))
      a_cl[n1..., n1...] = other.a - other.b.matmul(@d).matmul(e).matmul(other.c)
      # b_cl, c_cl, d_cl... (logica corretta per -1)
      # ...
      CrySpace::StateSpace.new(a_cl, b_cl, c_cl, d_cl)
    end
  end
end
