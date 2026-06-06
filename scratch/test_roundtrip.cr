require "../src/cryspace"

def test_roundtrip(num_arr, den_arr)
  tf = CrySpace::TransferFunction.new(num_arr.to_tensor, den_arr.to_tensor)
  puts "Original TF: num = #{tf.num}, den = #{tf.den}"
  ss = tf.to_statespace
  puts "StateSpace: A=\n#{ss.a}\nB=\n#{ss.b}\nC=\n#{ss.c}\nD=\n#{ss.d}"
  tf_back = ss.to_transferfunction
  puts "Recovered TF: num = #{tf_back.num}, den = #{tf_back.den}"
  puts "---"
end

test_roundtrip([1.0], [1.0, 1.0])
test_roundtrip([1.0, 2.0], [1.0, 3.0, 2.0])
test_roundtrip([51.5, 151.0, 100.0], [1.0, 100.0, 0.0])
