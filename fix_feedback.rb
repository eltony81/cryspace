content = File.read("src/cryspace/statespace.cr")
new_logic = <<-CRYSTAL
    def feedback(other : StateSpace, sign = -1)
      n1 = n_states
      n2 = other.n_states
      
      # E = inv(I + sign * D2 * D1)
      eye_outputs = Float64Tensor.identity(other.n_inputs)
      e = (eye_outputs + (other.d.matmul(@d)) * sign).inv
      
      a_cl = Float64Tensor.zeros([n1 + n2, n1 + n2])
      a_cl[0...n1, 0...n1] = @a - @b.matmul(e).matmul(other.d).matmul(@c) * sign
      a_cl[0...n1, n1...] = -@b.matmul(e).matmul(other.c) * sign
      a_cl[n1..., 0...n1] = other.b.matmul(@c - @d.matmul(e).matmul(other.d).matmul(@c))
      a_cl[n1..., n1...] = other.a - other.b.matmul(@d).matmul(e).matmul(other.c) * sign
      
      b_cl = Float64Tensor.zeros([n1 + n2, n_inputs])
      b_cl[0...n1, 0...] = @b.matmul(e)
      b_cl[n1..., 0...] = other.b.matmul(@d).matmul(e)
      
      c_cl = Float64Tensor.zeros([n_outputs, n1 + n2])
      c_cl[0..., 0...n1] = @c - @d.matmul(e).matmul(other.d).matmul(@c)
      c_cl[0..., n1...] = -@d.matmul(e).matmul(other.c)
      
      d_cl = @d.matmul(e)
      
      CrySpace::StateSpace.new(a_cl, b_cl, c_cl, d_cl)
    end
CRYSTAL

content.gsub!(/def feedback\(other : StateSpace, sign = -1\).*?end\n    end/m, new_logic)
File.write("src/cryspace/statespace.cr", content)
